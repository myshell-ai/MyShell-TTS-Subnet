from tts_rater.models import ReferenceEncoder
import torch
from tts_rater.mel_processing import spectrogram_torch
import librosa
import os
from easydict import EasyDict as edict
from torch.nn.functional import cosine_similarity, mse_loss
import glob
import numpy as np
import whisper
from jiwer import wer
import json
import random
script_dir = os.path.dirname(os.path.abspath(__file__))

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


def extract_se(ref_enc, ref_wav_list, se_save_path=None, device="cpu"):
    if isinstance(ref_wav_list, str):
        ref_wav_list = [ref_wav_list]

    gs = []

    for fname in ref_wav_list:
        audio_ref, sr = librosa.load(fname, sr=hps.data.sampling_rate)
        y = torch.FloatTensor(audio_ref)
        y = y.to(device)
        y = y.unsqueeze(0)
        y = spectrogram_torch(
            y,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        ).to(device)
        with torch.no_grad():
            g = ref_enc(y.transpose(1, 2)).unsqueeze(-1)
            gs.append(g.detach())
    gs = torch.stack(gs).mean(0)

    if se_save_path is not None:
        os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
        torch.save(gs.cpu(), se_save_path)

    return gs.squeeze(-1)


ref_enc = ReferenceEncoder(1024 // 2 + 1, 256)
checkpoint = torch.load(os.path.join(script_dir, "reference_encoder.pth"), map_location="cpu")
ref_enc.load_state_dict(checkpoint["model"], strict=True)
vec_gt_dict = torch.load(os.path.join(script_dir, "vec_gt.pth"), map_location="cpu")


def compute_tone_color_similarity(audio_paths, vec_gt):
    scores = []
    for wav_gen in audio_paths:
        vec_gen = extract_se(ref_enc, wav_gen)
        score = cosine_similarity(vec_gen, vec_gt).item()
        # in order to use the score as the loss, we use 1 - score
        scores.append(1 - score)
    return scores


model = whisper.load_model("medium")


def compute_wer(texts, audio_paths):
    wer_results = []
    assert len(texts) == len(audio_paths)
    for text, audio_path in zip(texts, audio_paths):
        result = model.transcribe(audio_path)
        # print(result)
        wer_results.append(wer(text.strip(), result["text"]))
    return wer_results


texts = json.load(open(os.path.join(script_dir, "text_list.json")))


def rate(ckpt_path, speaker="p225", seed=0, samples=10):
    from melo.api import TTS

    model = TTS(language="EN", device="auto", ckpt_path=ckpt_path)
    speaker_ids = model.hps.data.spk2id
    spkr = speaker_ids["EN-US"]

    random.seed(seed)
    text_test = random.choices(texts, k=samples)

    # remove the directory if it exists
    if os.path.exists("tmp"):
        os.system("rm -r tmp")

    for i, text in enumerate(text_test):
        save_path = f"tmp/sent_{i:03d}.wav"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.tts_to_file(text, spkr, save_path, speed=1.0)

    audio_paths = sorted(glob.glob("tmp/*.wav"))

    tone_color_sim = compute_tone_color_similarity(audio_paths, vec_gt_dict[speaker])
    word_error_rate = compute_wer(text_test, audio_paths)

    return tone_color_sim + word_error_rate


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)

    args = parser.parse_args()

    print(rate(args.ckpt_path))
