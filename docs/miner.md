# Miner

We use [MeloTTS](https://github.com/myshell-ai/MeloTTS), a lightweight yet performant TTS model, as the backbone of our TTS subnet. The miner is responsible for training the TTS model and submitting it to Huggingface 🤗. You can refer to the [training document](https://github.com/myshell-ai/MeloTTS/blob/main/docs/training.md) from the MeloTTS repo for the training guideline.

After training, please use the following command to submit the model to Huggingface 🤗:

```bash
python tts_subnet/upload_model.py --hf_repo_id huggingface_repo_name --load_model_dir path_to_your_checkpoint.pth_file --wallet.name your_wallet --wallet.hotkey your_hotkey
```
For example:

```bash
python tts_subnet/upload_model.py --hf_repo_id myshell-ai/melotts --load_model_dir /melo-en/checkpoint.pth --wallet.name myshell --wallet.hotkey shell
```

Please make sure you have added your huggingface API key to the `.env` file. For example:

```bash
HF_ACCESS_TOKEN="hf_YOUR_API_KEY"
```