# 词汇表辅助翻译器

[English](README.md) | 中文版（机翻）

一个可以使用词汇表翻译任何内容的工具箱。
- 针对字幕翻译优化
- 可选择本地托管模型或API
- 通过RAG驱动的词汇表生成实现低资源翻译
- （可能）自动字幕合并和拆分

你只需要提供一个词汇表文件。

## 待办事项
- [ ] 从项目中移除Langchain，功能太有限s
- [ ] 添加额外的ASR模型。Whisper和fast-whisper经常产生幻觉，给出没有大写字母的长文本
- [ ] 评估是否需要RAG系统来检索词汇
  - [ ] 纯字符串匹配算法似乎比RAG效果更好
  - [ ] 也许可以先使用字符串匹配，然后用LLM去除语义不相关的词汇？
- [ ] 改进提示以减少token使用
  - [ ] 重新格式化系统和用户提示
  - [ ] 利用API提供商的提示缓存
- [ ] 添加对vllm或Llama.cpp的支持，因为它们允许运行比Ollama更多的模型（低优先级）

## 安装

假设你有一个NVIDIA GPU并且已经安装了NVIDIA驱动。如果没有，你仍然可以使用API来完成本项目中的大多数任务。我在Windows子系统Linux下的Ubuntu上运行。你的设置可能会有所不同。

> 这将更改为移除langchain。

```bash
sudo apt update -y 
sudo apt upgrade -y 
sudo apt insall ffmpeg

conda create -n video-translation python=3.12.4
conda activate video-translation
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install faster_whisper
conda install cudnn # 你可能需要这个来运行faster_whisper
pip install transformers
pip install langchain langchain_community langchain_chroma # 我肯定忘记了一些
```

如果你想使用API，你需要安装以下包。

```bash
pip install openai
```

## 词汇表文件格式

你应该把所有词汇表文件放在一个目录中。默认是`data`目录。你可以选择将所有词汇放在一个文件中，或者将它们分开以便更好地管理。

无论你决定如何存储它们，它们都应该是`csv`文件。每个csv文件应包含4列或更多列。必需的4列是`Term`、`Translation`、`Definition`、`Example`。你可以使用其他列作为元数据。

这是一个示例文件。

```csv
"Term","Translation","Definition","Example"
"Ollama","Ollama","Ollama是一个可以让你一键在本地运行LLM的软件。","I prefer Ollama over vllm because it is simple. --> 相比vllm，我还是更喜欢Ollama的简洁。"
"Whisper","Whisper","Whisper是openAI开发的ASR模型","Whisper is a ASR model develoepd by openAI. --> Whisper是一个由openAI开发的自动语言识别模型。"
```

## 类似项目

在开发这个项目时，我发现了一些类似的项目。
它们有更好的UI，目前功能也更完善。
- [RSS-Translator](https://github.com/rss-translator/RSS-Translator)
- [video-subtitle-master](video-subtitle-master)

然而，我的项目旨在开发一个低资源翻译系统，可以用于翻译小众主题。

## 吐槽

- 这个项目起源于我想翻译一些视频到中文。现在，它变得一团糟。
- DeepSeek API太便宜了。它也比我能在笔记本电脑上运行的任何模型都要好。我开始认为我应该直接使用API而不是Ollama。
