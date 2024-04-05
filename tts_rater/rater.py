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
script_dir = os.path.dirname(os.path.abspath(__file__))

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required to run a validator on this subnet")

# Batch helper
def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(itertools.islice(it, n))):
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
    batch = [torch.narrow(y, 0, 0, min(y.size(0), max_len)) for y in batch] # truncate overly-long inputs
    batch = [torch.nn.functional.pad(y, (0, max_len - y.size(0)), value=0) for y in batch]
    return batch

def extract_se(ref_enc, waveforms, batch_size, se_save_path=None):
    gs = []

    for y in tqdm(batched(waveforms, batch_size), total=math.ceil(len(waveforms)/batch_size)):
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
checkpoint = torch.load(os.path.join(script_dir, "reference_encoder.pth"), map_location="cuda")
ref_enc.load_state_dict(checkpoint["model"], strict=True)
vec_gt_dict = torch.load(os.path.join(script_dir, "vec_gt.pth"), map_location="cuda")

def compute_tone_color_similarity(audio_paths, vec_gt, batch_size):
    waveforms = [load_wav_file(fname, hps.data.sampling_rate) for fname in audio_paths]
    vec_gen = extract_se(ref_enc, waveforms, batch_size)
    scores = cosine_similarity(vec_gen, vec_gt)
    # in order to use the score as the loss, we use 1 - score
    scores = 1 - scores
    return scores.cpu().tolist()


whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").cuda()

def compute_wer(texts, audio_paths, batch_size):
    waveforms = [load_wav_file(fname, 16000) for fname in audio_paths]
    wer_results = []
    assert len(texts) == len(waveforms)
    for text_batch, audio_batch in tqdm(zip(batched(texts, batch_size), batched(waveforms, batch_size)), total=math.ceil(len(waveforms)/batch_size)):
        audio_batch = pad_audio_batch(audio_batch, max_len=16000*30)
        audio_batch = [audio.numpy() for audio in audio_batch]
        inputs = whisper_processor(
            audio=audio_batch,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
            sampling_rate=16000
        )
        inputs['input_features'] = inputs["input_features"].cuda()
        generated_ids = whisper_model.generate(**inputs)
        transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)
        for in_text, out_text in zip(text_batch, transcription):
            wer_results.append(wer(in_text.strip(), out_text.strip()))
    return wer_results


texts = json.load(open(os.path.join(script_dir, "text_list.json")))


def rate(ckpt_path, speaker="p225", seed=0, samples=100, batch_size=16):
    from melo.api import TTS

    model = TTS(language="EN", device="cuda", ckpt_path=ckpt_path)
    speaker_ids = model.hps.data.spk2id
    spkr = speaker_ids["EN-US"]

    random.seed(seed)
    text_test = random.choices(texts, k=samples)

    if os.path.exists("tmp"):
        os.system("rm -r tmp")

    for i, text in enumerate(tqdm(text_test)):
        save_path = f"tmp/sent_{i:03d}.wav"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.tts_to_file(text, spkr, save_path, speed=1.0, quiet=True)

    audio_paths = sorted(glob.glob("tmp/*.wav"))

    tone_color_sim = compute_tone_color_similarity(audio_paths, vec_gt_dict[speaker], batch_size)
    word_error_rate = compute_wer(text_test, audio_paths, batch_size)

    return tone_color_sim + word_error_rate


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)

    args = parser.parse_args()

    print(rate(args.ckpt_path))
