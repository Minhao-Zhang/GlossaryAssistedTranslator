# Glossary Assisted Translator

[中文版](README_zh.md) | English

A tool box that can be used to translate anything with glossary. 
- optimized for subtitle translation
- choose between local hosted model or APIs
- low resource translation with glossary assistance
- (potentially) automatic subtitle merging and breaking

All you need to bring is a glossary file. 


## Installation

I am assuming you have a NVIDIA GPU and you have installed the NVIDIA drivers. If not, you can still use APIs to accomplish most tasks in this project. I am running on Ubuntu under Windows Subsystem Linux. Your set up might be a bit different. 

You can use pretty much any modern version of python, tested on `python 3.10.12` and `python 3.12.4`. 
```bash
sudo apt update -y 
sudo apt upgrade -y 
sudo apt insall ffmpeg

conda create -n gat python=3.10.12
pip install -r requirements.txt
conda install cudnn # you might need this for faster-whisper
```

This will set up a environment for running the ASR model locally and calling any LLM APIs.
If you want all local translation, install `Ollama` and pull whatever LLM model you want to use. 
A good starting model is `qwen2.5` as it has a wide language support and it has a series of model size.

## Glossary File Format

### Glossary Matcher Format

You should put all your glossary file(s) in a directory. The default is `data`. You can choose to place all your glossary in one file or split them up to better manage them.

Regardless on how you decides to store them, they all should be `csv` files. Each csv file shall contain 4 required columns. The required 4 are `Term`, `Translation`, `Definition`, `Example`. You use any other column as metadata.

Here is a sample file.

```csv
"Term","Translation","Definition","Example"
"Ollama","Ollama","Ollama is a software that will enable you to run LLMs locally with one-click.","I prefer Ollama over vllm because it is simple. --> 相比vllm，我还是更喜欢Ollama的简洁。"
"Whisper","Whisper","Whisper is a ASR model developed by openAI","Whisper is a ASR model develoepd by openAI. --> Whisper是一个由openAI开发的自动语言识别模型。"
```

## Quick Start 

See [example.ipynb](example.ipynb). 


## Similar Projects 

In developing this project, I have found some projects that provides a good UI for translating subtitles into another language.
- [RSS-Translator](https://github.com/rss-translator/RSS-Translator)
- [video-subtitle-master](video-subtitle-master)

## Rant 

- This project started off that I want to translate some videos into Chinese. Now, I wish to build this into an actual toolbox.
- DeepSeek API is so cheap. It is also way better than any model I can host on my laptop. I started to think that I should just use the API instead of Ollama.
- RIP DeepSeek server being DDOSed.