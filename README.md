# Open Video Translator

A tool box that can be used to translate videos. 
- optimized for subtitle translation
- choose between local hosted model or APIs
- low-resource translation with RAG-powered glossary generation
- (potentially) automatic subtitle merging and breaking

## TODO 
- [x] Build a translation agent with glossary support
  - [x] Glossary collection
  - [x] Glossary selection powered by LLM
  - [x] Translation using selected glossary
- [ ] Build a proper glossary retrival system
  - [x] Use Langchain to build a RAG foundation class
  - [x] Allow more embedding models to be used
  - [ ] Fine tune embedding prompt and retrieval prompt
  - [x] Build a basic nltk-based glossary retrival system
  - [ ] Collect some more glossary with better translation
  - [ ] Improve Zero-Shot or Few-Shot Prompting using these translations
  - [ ] Experiment with different similarity function for RAG
  - [ ] Align RAG glossary with NLTK based glossary
- [ ] Use Llama.cpp instead of Ollama to run inference models as it supports almost any models you can find on Huggingface
- [ ] Add support for using different APIs

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

If you want to use APIs, you need to install the following packages. 

```bash
pip install openai
```

## Similar Projects 

In developing this project, I have found some similar projects.
They have much better UI and is fully functional at the moment. 
- [RSS-Translator](https://github.com/rss-translator/RSS-Translator)
- [video-subtitle-master](video-subtitle-master)

However, my project aims to develop a low-resource translation system that can used to translated niche topics. 

## Rant 

DeepSeek API is so cheap. It is also way better than any of the local hosted models. I started to think that I should just use the API instead of Ollama.