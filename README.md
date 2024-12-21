# Open Video Translator

A tool box that can be used to download and translate almost any video to any language. 
It will be all **local** and **open-source**. 
With LLMs supporting long context, translation of each line of subtitle will be context aware. 

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) is used to download the video ([supported sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)). 
- [openai-whisper](https://github.com/openai/whisper) is used to transcribe the video ([supported languages](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py)). 
- [ollama](https://github.com/ollama/ollama-python) is used to translate the video. Supported languages can vary from the model you chose. You shall refer to the respective model page for the supported languages. 

## TODO 
- Use Transformers package instead of ollama.

## Installation

I am assuming you have a NVIDIA GPU and you have installed the NVIDIA drivers. 
I am running on Ubunbu under Windows Subsystem Linux, so my set up might be a bit different. 

```bash
sudo apt update -y 
sudo apt upgrade -y 
sudo apt insall ffmpeg

conda create -n video-translation python=3.12.4
conda activate video-translation
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install faster_whisper
conda install cudnn # you might need this for faster_whisper
conda isntall transformers 
```
