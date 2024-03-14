---
license: mit
language:
- ko
pipeline_tag: text-to-speech
---

# MeloTTS

MeloTTS is a **high-quality multi-lingual** text-to-speech library by [MyShell.ai](https://myshell.ai). Supported languages include:


| Model card | Example |
| --- | --- |
| [English](https://huggingface.co/myshell-ai/MeloTTS-English-v2) (American)    | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/en/EN-US/speed_1.0/sent_000.wav) |
| [English](https://huggingface.co/myshell-ai/MeloTTS-English-v2) (British)     | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/en/EN-BR/speed_1.0/sent_000.wav) |
| [English](https://huggingface.co/myshell-ai/MeloTTS-English-v2) (Indian)      | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/en/EN_INDIA/speed_1.0/sent_000.wav) |
| [English](https://huggingface.co/myshell-ai/MeloTTS-English-v2) (Australian)  | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/en/EN-AU/speed_1.0/sent_000.wav) |
| [English](https://huggingface.co/myshell-ai/MeloTTS-English-v2) (Default)     | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/en/EN-Default/speed_1.0/sent_000.wav) |
| [Spanish](https://huggingface.co/myshell-ai/MeloTTS-Spanish)               | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/es/ES/speed_1.0/sent_000.wav) |
| [French](https://huggingface.co/myshell-ai/MeloTTS-French)                | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/fr/FR/speed_1.0/sent_000.wav) |
| [Chinese](https://huggingface.co/myshell-ai/MeloTTS-Chinese) (mix EN)      | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/zh/ZH/speed_1.0/sent_008.wav) |
| [Japanese](https://huggingface.co/myshell-ai/MeloTTS-Japanese)              | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/jp/JP/speed_1.0/sent_000.wav) |
| [Korean](https://huggingface.co/myshell-ai/MeloTTS-Korean/)                | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/kr/KR/speed_1.0/sent_000.wav) |

Some other features include:
- The Chinese speaker supports `mixed Chinese and English`.
- Fast enough for `CPU real-time inference`.


## Usage

### Without Installation

An unofficial [live demo](https://huggingface.co/spaces/mrfakename/MeloTTS) is hosted on Hugging Face Spaces.

#### Use it on MyShell

There are hundreds of TTS models on MyShell, much more than MeloTTS. See examples [here](https://github.com/myshell-ai/MeloTTS/blob/main/docs/quick_use.md#use-melotts-without-installation).
More can be found at the widget center of [MyShell.ai](https://app.myshell.ai/robot-workshop).

### Install and Use Locally

Follow the installation steps [here](https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md#linux-and-macos-install) before using the following snippet:

```python
from melo.api import TTS

# Speed is adjustable
speed = 1.0

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = 'auto' # Will automatically use GPU if available

# English 
text = "Did you ever hear a folk tale about a giant turtle?"
model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id

# American accent
output_path = 'en-us.wav'
model.tts_to_file(text, speaker_ids['EN-US'], output_path, speed=speed)

# British accent
output_path = 'en-br.wav'
model.tts_to_file(text, speaker_ids['EN-BR'], output_path, speed=speed)

# Indian accent
output_path = 'en-india.wav'
model.tts_to_file(text, speaker_ids['EN_INDIA'], output_path, speed=speed)

# Australian accent
output_path = 'en-au.wav'
model.tts_to_file(text, speaker_ids['EN-AU'], output_path, speed=speed)

# Default accent
output_path = 'en-default.wav'
model.tts_to_file(text, speaker_ids['EN-Default'], output_path, speed=speed)

```


## Join the Community

**Open Source AI Grant**

We are actively sponsoring open-source AI projects. The sponsorship includes GPU resources, fundings and intellectual support (collaboration with top research labs). We welcome both reseach and engineering projects, as long as the open-source community needs them. Please contact [Zengyi Qin](https://www.qinzy.tech/) if you are interested.

**Contributing**

If you find this work useful, please consider contributing to the GitHub [repo](https://github.com/myshell-ai/MeloTTS).

- Many thanks to [@fakerybakery](https://github.com/fakerybakery) for adding the Web UI and CLI part.

## License

This library is under MIT License, which means it is free for both commercial and non-commercial use.

## Acknowledgements

This implementation is based on [TTS](https://github.com/coqui-ai/TTS), [VITS](https://github.com/jaywalnut310/vits), [VITS2](https://github.com/daniilrobnikov/vits2) and [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2). We appreciate their awesome work.

