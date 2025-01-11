# Glossary Assisted Translator

[中文版](README_zh.md) | English

A tool box that can be used to translate anything with glossary. 
- optimized for subtitle translation
- choose between local hosted model or APIs
- low-resource translation with RAG-powered glossary generation
- (potentially) automatic subtitle merging and breaking

All you need to bring is a glossary file. 

## TODO 
- [ ] Add additional ASR models. Whisper and fast-whisper often hullucinates and give me long text without captalization and 
- [x] Assess te need for RAG system at all in retriving words. 
  - [x] Using a purely string matching algorithm seems to work better than RAG. 
  - [x] Perhaps a string matching, then use LLM to remove semantically irrelevent terms?
- [x] Improve the prompting techniques sto reduce token usage. 
  - [x] Re-format the system and user prompt. 
  - [x] Take advantage of prompt caching from API providers.
- [ ] Add support for vllm or Llama.cpp as they allow you to run more models than Ollama. (Low priority)


## Installation

I am assuming you have a NVIDIA GPU and you have installed the NVIDIA drivers. If not, you can still use APIs to accomplish most tasks in this project. I am running on Ubunbu under Windows Subsystem Linux. Your set up might be a bit different. 


```bash
sudo apt update -y 
sudo apt upgrade -y 
sudo apt insall ffmpeg

conda create -n gat python=3.12.4
pip install -r requirements.txt
```

If you want to use APIs, you need to install the corresponding python package. 

```bash
pip install openai
```

## Glossary File Format 

You should put all your glossary file(s) in a directory. The default is `data`. You can choose to place all your glossary in one file or split them up to better manage them. 

Regardless on how you decides to store them, they all should be `csv` files. Each csv file shall contain 4 required columns. The required 4 are `Term`, `Translation`, `Definition`, `Example`. You use any other column as metadata. 

Here is a sample file. 

```csv
"Term","Translation","Definition","Example"
"Ollama","Ollama","Ollama is a software that will enable you to run LLMs locally with one-click.","I prefer Ollama over vllm because it is simple. --> 相比vllm，我还是更喜欢Ollama的简洁。"
"Whisper","Whisper","Whisper is a ASR model developed by openAI","Whisper is a ASR model develoepd by openAI. --> Whisper是一个由openAI开发的自动语言识别模型。"
```

## Similar Projects 

In developing this project, I have found some projects that provides a good UI for translating subtitles into another language.
- [RSS-Translator](https://github.com/rss-translator/RSS-Translator)
- [video-subtitle-master](video-subtitle-master)

## Rant 

- This project started off that I want to translate some videos into Chinese. Now, I wish to build this into an actual toolbox.
- DeepSeek API is so cheap. It is also way better than any model I can host on my laptop. I started to think that I should just use the API instead of Ollama.
