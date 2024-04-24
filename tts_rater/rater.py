from whisper.normalizers import EnglishTextNormalizer

from tts_rater.models import ReferenceEncoder
import torch
from tts_rater.mel_processing import spectrogram_torch
import librosa
import os
import eng_to_ipa as ipa
from easydict import EasyDict as edict
from torch.nn.functional import cosine_similarity, mse_loss
import glob
import numpy as np
from jiwer import process_words
import json
import random
import math
import itertools
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm
import onnxruntime as ort
import tempfile

from tts_rater.pann import PANNModel

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

whisper_model = "openai/whisper-small.en"
whisper_processor = WhisperProcessor.from_pretrained(whisper_model)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model).cuda()
whisper_normalizer = EnglishTextNormalizer()


def compute_wer(texts, audio_paths, batch_size):
    waveforms = [load_wav_file(fname, 16000) for fname in audio_paths]

    total_errs = []
    total_words = []

    # wer_results = []
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
            in_text_whisper = whisper_normalizer(in_text)
            out_text_whisper = whisper_normalizer(out_text)
            output = process_words(in_text_whisper, out_text_whisper)

            for alignment in output.alignments[0]:
                if alignment.type != "substitute":
                    continue
                orig_word = in_text_whisper.split(" ")[alignment.ref_start_idx : alignment.ref_end_idx]
                pred_word = out_text_whisper.split(" ")[alignment.hyp_start_idx : alignment.hyp_end_idx]
                if ipa.convert(orig_word) == ipa.convert(pred_word):
                    output.substitutions -= 1
                    output.hits += 1

            S, D, I, H = output.substitutions, output.deletions, output.insertions, output.hits
            total_errs.append(I + S + D)
            total_words.append(S + D + H)

    return np.array(total_errs), np.array(total_words)


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


# =================== PANN MMD loss ===================
speaker_pann_embeds = torch.load(os.path.join(script_dir, "pann/pann_embeds.pth"), map_location="cuda")
pann_model = PANNModel()

def compute_mmd(a_x: torch.Tensor, b_y: torch.Tensor):
    _SIGMA = 10
    _SCALE = 1000

    a_x = a_x.double()
    b_y = b_y.double()

    a_x_sqnorms = torch.sum(a_x**2, dim=1)
    b_y_sqnorms = torch.sum(b_y**2, dim=1)

    gamma = 1 / (2 * _SIGMA**2)

    k_xx = torch.mean(torch.exp(-gamma * (-2 * (a_x @ a_x.T) + a_x_sqnorms[:, None] + a_x_sqnorms[None, :])))
    k_xy = torch.mean(torch.exp(-gamma * (-2 * (a_x @ b_y.T) + a_x_sqnorms[:, None] + b_y_sqnorms[None, :])))
    k_yy = torch.mean(torch.exp(-gamma * (-2 * (b_y @ b_y.T) + b_y_sqnorms[:, None] + b_y_sqnorms[None, :])))

    return _SCALE * (k_xx + k_yy - 2 * k_xy)


def compute_pann_mmd_loss(audio_paths: list[str], n_boostrap: int, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng()

    n_samples = len(audio_paths)
    waveforms = [load_wav_file(fname, 32000) for fname in audio_paths]

    embeddings = []
    for audio in tqdm(waveforms):
        audio = torch.Tensor(audio).cuda()
        embedding = pann_model.get_embedding(audio[None])[0]
        embeddings.append(embedding)
    embeddings = torch.stack(embeddings, dim=0)

    mmd_losses = []
    for ii in range(n_boostrap):
        idxs = rng.choice(n_samples, size=(n_samples,), replace=True)
        sampled_embeddings = embeddings[idxs]
        mmd = compute_mmd(speaker_pann_embeds, sampled_embeddings)
        mmd_losses.append(mmd.item())

    return mmd_losses


# =================== Rate function ===================

# TODO: Read texts from Internet or use a larger dataset
texts = json.load(open(os.path.join(script_dir, "text_list.json")))


def get_normalized_scores(raw_errs: dict[str, float]):
    score_ranges = {"pann_mmd": (0.0, 200.0), "word_error_rate": (0.0, 0.08), "tone_color": (0.15, 0.4)}
    normalized_scores = {}
    for key, value in raw_errs.items():
        min_val, max_val = score_ranges[key]
        normalized_err = (np.asarray(value) - min_val) / (max_val - min_val)
        normalized_scores[key] = np.clip(1 - normalized_err, 1e-6, 1.0)
    return normalized_scores


def compute_sharpe_ratios(scores: list[float]) -> list[float]:
    # Jackknife estimate of the Sharpe ratio.
    n = len(scores)
    sharpe_ratios = []
    for ii in range(n):
        scores_jack = scores[:ii] + scores[ii + 1 :]
        mean_jack = np.mean(scores_jack)
        std_jack = np.std(scores_jack, ddof=1)
        sharpe_jack = mean_jack / std_jack

        if mean_jack < 1e-6 and std_jack == 0.0:
            sharpe_jack = 0.0

        sharpe_ratios.append(sharpe_jack)

    return sharpe_ratios


def rate(
    ckpt_path,
    speaker="p225",
    seed=0,
    samples=64,
    batch_size=16,
    n_bootstrap: int = 256,
    use_tmpdir=False,
):
    return rate_(ckpt_path, speaker, seed, samples, batch_size, n_bootstrap, use_tmpdir)[0]


def rate_(
    ckpt_path,
    speaker="p225",
    seed=0,
    samples=64,
    batch_size=16,
    n_bootstrap: int = 256,
    use_tmpdir=False,
):
    """
    Compute the following metrics for a given checkpoint:
    - PANN MMD
    - Word error rate
    Then normalize the scores to (nominally) [0, 1] range.
    """
    from melo.api import TTS

    model = TTS(language="EN", device="cuda", ckpt_path=ckpt_path)
    speaker_ids = model.hps.data.spk2id
    spkr = speaker_ids["EN-US"]

    random.seed(seed)
    text_test = random.choices(texts, k=samples)
    rng = np.random.default_rng(seed=seed + 7)
    torch.random.manual_seed(seed + 11)

    with tempfile.TemporaryDirectory() as tmpdir:
        if use_tmpdir:
            tmpdir = "tmp"
            os.makedirs(tmpdir, exist_ok=True)
        for i, text in enumerate(tqdm(text_test)):
            save_path = os.path.join(tmpdir, f"{i:03d}.wav")
            model.tts_to_file(text, spkr, save_path, speed=1.0, quiet=True)

        audio_paths = sorted(glob.glob(os.path.join(tmpdir, "*.wav")))

        pann_mmds = compute_pann_mmd_loss(audio_paths, n_bootstrap, rng=rng)
        total_errs, total_words = compute_wer(text_test, audio_paths, batch_size)
        word_error_rates = []
        for _ in range(n_bootstrap):
            idxs = rng.choice(samples, (samples,), replace=True)
            word_error_rates.append(total_errs[idxs].sum() / total_words[idxs].sum())

        vec_gt = vec_gt_dict[speaker]
        tcs = compute_tone_color_loss(audio_paths, vec_gt, batch_size)
        idxs = rng.choice(samples, (n_bootstrap,), replace=True)
        tcs = np.asarray(tcs)[idxs]

    assert len(pann_mmds) == len(word_error_rates) == n_bootstrap
    raw_errs = {"pann_mmd": pann_mmds, "word_error_rate": word_error_rates, "tone_color": tcs}
    norm_dict = get_normalized_scores(raw_errs)

    keys = list(norm_dict.keys())
    norm_scores = []
    for ii in range(n_bootstrap):
        norm_score = np.prod([norm_dict[k][ii] for k in keys])
        norm_scores.append(norm_score)

    return norm_scores, norm_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--use_tmpdir", action="store_true")
    parser.add_argument("--speaker", type=str, default="p225")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--n_bootstrap", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    print(
        rate(
            args.ckpt_path,
            args.speaker,
            args.seed,
            args.samples,
            args.batch_size,
            args.n_bootstrap,
            args.use_tmpdir,
        )
    )
