# Open Video Translator

A tool box that can be used to translate videos. 
- optimized for subtitle translation
- no paid APIs, all you need is your laptop
- low-resource translation with RAG-powered glossary generation
- (potentially) automatic subtitle merging and breaking

## TODO 
- [x] Build a translation agent with glossary injection
  - [x] Glossary collection
  - [x] Glossary selection powered by LLM
  - [x] Translation using selected glossary
- [ ] Build a proper RAG system for glossary retrival
  - [x] Use Langchain to build a RAG foundation class
  - [ ] Collect some more glossary with better translation
  - [ ] Build Few-Shot Prommpting using these translations
  - [ ] Allow more embedding models to be used
  - [ ] Experiment with different similarity function
- [ ] Use Llama.cpp instead of Ollama to run inference models as it supports almost any models you can find on Huggingface

## Installation

I am assuming you have a NVIDIA GPU and you have installed the NVIDIA drivers. 
I am running on Ubunbu under Windows Subsystem Linux. Your set up might be a bit different. 

```bash
sudo apt update -y 
sudo apt upgrade -y 
sudo apt insall ffmpeg

conda create -n video-translation python=3.12.4
conda activate video-translation
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install faster_whisper
conda install cudnn # you might need this for faster_whisper
pip install transformers
pip install langchain langchain_community langchain_chroma # I certainly forgot some of them
```
