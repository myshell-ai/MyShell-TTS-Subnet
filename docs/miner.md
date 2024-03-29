# Miner

We use [MeloTTS](https://github.com/myshell-ai/MeloTTS), a lightweight yet performant TTS model, as the backbone of our TTS subnet. The miner is responsible for training the TTS model and submitting it to Huggingface 🤗. You can refer to the [training document](https://github.com/myshell-ai/MeloTTS/blob/main/docs/training.md) from the MeloTTS repo for the training guideline. We will also provide a detailed training setup here. But please keep in mind -- this is just a reference setup, and the game will become super competitive, you will likely need to develop your own secret sauce to train the best model.

## Training Pipeline
### 1. Install the dependencies
Please make sure you have installed the package following `README.md` in the root directory.

### 2. Prepare the dataset
We use the [VCTK](https://huggingface.co/datasets/vctk) dataset as the source of our speaker data for now. And the goal is to build a model that can mimic a specific speaker's voice. You can download the dataset from the Huggingface website and extract the speaker's data you want to train on. For example, you can extract the speaker's data with the id `p225`. You can use `librosa` to convert the audio files to the `wav` format and put them in a folder.

Then you can create a `metadata.list` file in the folder you want to store all the configuration files. Each line of the file should be in the following format:

```bash
Path to your .wav file|EN-US|EN|The text associated with the audio file
```
where `EN-US|EN` should be untouched since we will evaluate under this configuration.

Then, you can run
```bash
python preprocess_text.py --metadata path_to_your_metadata.list 
```
to get preprocessed configs and data.

### 3. Train the model
We provide a script in `train.sh` to train the model. But you can also run the following command to train the model:

```bash
torchrun --nproc_per_node=your_num_gpus --master_port=your_port \
    train.py --c path_to_your_config --model the_model_name_you_want_to_store 
```
where the `config.json` will be generated in the same folder as the `metadata.list` file. You can modify the `config.json` file to adjust the training hyperparameters. For example, you can change the `batch_size`, `num_workers`, `lr`, `max_steps`, etc.

In addition, you can add the `--pretrain_G path_to_your_pretrained_model_pth_file` to load the pretrained model. For example, the official MeloTTS model can be downloaded from the [Huggingface model hub](https://huggingface.co/myshell-ai/MeloTTS-English).

> Warning: As for now, there is an incompatible issue between the pretrained model and the default model configuration. We've fixed it in the newest version. But if you are using an old version of this package, please edit the `config.json` file in the same folder as the `metadata.list` file and change the `n_speakers` in the `data` section from 1 to 256 and add an additional line `"num_languages": 10` in the model section. An example of a good configuration file is provided in the `docs/config.json`. Sorry for the inconvenience.

### 4. Listen to the generated audio
After training, you can run the following command to generate audio from the text:

```bash
python infer.py --text "<some text here>" -m path_to_your_G_<iter>.pth -o <output_dir>
```
where the `G_<iter>.pth` is the model checkpoint which will be saved under the `logs/your_model_name/` folder. **This is also the file you need to submit to Huggingface 🤗.**


## Submitting to Bittensor

After training, please use the following command to submit the model to Huggingface 🤗 (assuming `tts_subnet` package has been installed):

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

Since MeloTTS is a lightweight model, you can train it on a single consumer-grade GPU.