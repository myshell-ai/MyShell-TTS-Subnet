from tts_rater.models import ReferenceEncoder
import torch
from tts_rater.mel_processing import spectrogram_torch
import librosa
import os
from easydict import EasyDict as edict
from torch.nn.functional import cosine_similarity, mse_loss
import glob
import numpy as np
from jiwer import wer
import json
import random
import math
import itertools
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm
import onnxruntime as ort
import tempfile

script_dir = os.path.dirname(os.path.abspath(__file__))

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required to run a validator on this subnet")


# Batch helper
def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


hps = edict(
    {
        "data": {
            "sampling_rate": 22050,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
        }
    }
)


def load_wav_file(fname, new_sample_rate):
    audio_ref, sr = librosa.load(fname, sr=new_sample_rate)
    y = torch.FloatTensor(audio_ref)
    return y


def pad_audio_batch(batch, max_len=0):
    if not max_len:
        max_len = max(map(len, batch))
    batch = [
        torch.narrow(y, 0, 0, min(y.size(0), max_len)) for y in batch
    ]  # truncate overly-long inputs
    batch = [
        torch.nn.functional.pad(y, (0, max_len - y.size(0)), value=0) for y in batch
    ]
    return batch


# =================== Tone color loss ===================


def extract_se(ref_enc, waveforms, batch_size, se_save_path=None):
    gs = []

    for y in tqdm(
        batched(waveforms, batch_size), total=math.ceil(len(waveforms) / batch_size)
    ):
        y = pad_audio_batch(y)
        y = torch.stack(y)
        y = y.cuda()

        y = spectrogram_torch(
            y,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        with torch.no_grad():
            g = ref_enc(y.transpose(1, 2))
            gs.append(g.detach())
    gs = torch.cat(gs)

    if se_save_path is not None:
        os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
        torch.save(gs.cpu(), se_save_path)

    return gs


ref_enc = ReferenceEncoder(1024 // 2 + 1, 256).cuda()
checkpoint = torch.load(
    os.path.join(script_dir, "reference_encoder.pth"), map_location="cuda"
)
ref_enc.load_state_dict(checkpoint["model"], strict=True)
vec_gt_dict = torch.load(os.path.join(script_dir, "vec_gt.pth"), map_location="cuda")


def compute_tone_color_loss(audio_paths, vec_gt, batch_size):
    waveforms = [load_wav_file(fname, hps.data.sampling_rate) for fname in audio_paths]
    vec_gen = extract_se(ref_enc, waveforms, batch_size)
    sims = cosine_similarity(vec_gen, vec_gt)
    # in order to use the score as the loss, we use 1 - score
    losses = 1 - sims
    return losses.cpu().tolist()


# =================== Word error rate ===================

whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small.en"
).cuda()


def compute_wer(texts, audio_paths, batch_size):
    waveforms = [load_wav_file(fname, 16000) for fname in audio_paths]
    wer_results = []
    assert len(texts) == len(waveforms)
    for text_batch, audio_batch in tqdm(
        zip(batched(texts, batch_size), batched(waveforms, batch_size)),
        total=math.ceil(len(waveforms) / batch_size),
    ):
        audio_batch = pad_audio_batch(audio_batch, max_len=16000 * 30)
        audio_batch = [audio.numpy() for audio in audio_batch]
        inputs = whisper_processor(
            audio=audio_batch,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
            sampling_rate=16000,
        )
        inputs["input_features"] = inputs["input_features"].cuda()
        generated_ids = whisper_model.generate(**inputs)
        transcription = whisper_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        for in_text, out_text in zip(text_batch, transcription):
            wer_results.append(wer(in_text.strip(), out_text.strip()))
    return wer_results


# =================== DNS-MOS loss ===================
"""Adapted from https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS"""

INPUT_LENGTH = 9

# Only use p808_onnx_sess for now
# primary_model = ort.InferenceSession(os.path.join(script_dir, "DNSMOS/sig_bak_ovr.onnx"))
p808_onnx_sess = ort.InferenceSession(os.path.join(script_dir, "DNSMOS/model_v8.onnx"))


def load_melspecs(fname):
    audio_ref, sr = librosa.load(fname, sr=16000)
    len_samples = int(INPUT_LENGTH * sr)
    # Repeat or truncate the audio to the desired length
    if len(audio_ref) < len_samples:
        audio_ref = np.tile(audio_ref, 1 + int(np.ceil(len_samples / len(audio_ref))))
    audio_ref = audio_ref[:len_samples]
    mel_spec = librosa.feature.melspectrogram(
        y=audio_ref, sr=16000, n_fft=321, hop_length=160, n_mels=120
    )
    # to_db
    mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
    return mel_spec


def compute_dns_mos_loss(audio_paths, batch_size):
    mel_specs = [np.array(load_melspecs(f).T).astype("float32") for f in audio_paths]
    dns_mos_results = []
    for mel_batch in tqdm(
        batched(mel_specs, batch_size), total=math.ceil(len(mel_specs) / batch_size)
    ):
        p808_oi = {"input_1": np.stack(mel_batch)}
        outputs = p808_onnx_sess.run(None, p808_oi)
        dns_mos_results.extend(outputs[0].reshape(-1).tolist())
    # convert to loss
    return [-x for x in dns_mos_results]


# =================== Rate function ===================

# TODO: Read texts from Internet or use a larger dataset
texts = json.load(open(os.path.join(script_dir, "text_list.json")))


def rate(
    ckpt_path,
    speaker="p225",
    seed=0,
    samples=64,
    batch_size=16,
    group_size=16,
    use_tmpdir=False,
):
    """
    Compute the following metrics for a given checkpoint:
    - Tone color loss
    - Word error rate
    And then aggregate the losses by group_size to improve the stability of the results.

    """
    from melo.api import TTS

    model = TTS(language="EN", device="cuda", ckpt_path=ckpt_path)
    speaker_ids = model.hps.data.spk2id
    spkr = speaker_ids["EN-US"]

    random.seed(seed)
    text_test = random.choices(texts, k=samples)

    with tempfile.TemporaryDirectory() as tmpdir:
        if use_tmpdir:
            tmpdir = "tmp"
            os.makedirs(tmpdir, exist_ok=True)
        for i, text in enumerate(tqdm(text_test)):
            save_path = os.path.join(tmpdir, f"{i:03d}.wav")
            model.tts_to_file(text, spkr, save_path, speed=1.0, quiet=True)

        audio_paths = sorted(glob.glob(os.path.join(tmpdir, "*.wav")))

        tone_color_losses = compute_tone_color_loss(
            audio_paths, vec_gt_dict[speaker], batch_size
        )
        word_error_rate = compute_wer(text_test, audio_paths, batch_size)
        dns_mos_losses = compute_dns_mos_loss(audio_paths, batch_size)

    losses = tone_color_losses + word_error_rate + dns_mos_losses

    # Aggregate the losses by group_size
    agg_losses = []
    for i in range(0, len(losses), group_size):
        agg_losses.append(sum(losses[i : i + group_size]) / group_size)

    return agg_losses


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--use_tmpdir", action="store_true")
    parser.add_argument("--speaker", type=str, default="p225")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=16)
    args = parser.parse_args()

    print(
        rate(
            args.ckpt_path,
            args.speaker,
            args.seed,
            args.samples,
            args.batch_size,
            args.group_size,
            args.use_tmpdir,
        )
    )
