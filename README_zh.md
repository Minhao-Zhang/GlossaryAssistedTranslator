# 词汇表辅助翻译器

[English](README.md) | 中文版（机翻且可能过时了）

一个可以使用词汇表翻译任何内容的工具箱。
- 针对字幕翻译优化
- 可选择本地托管模型或API
- 通过RAG驱动的词汇表生成实现低资源翻译
- （可能）自动字幕合并和拆分

你只需要提供一个词汇表文件。

## 待办事项
- [ ] 添加额外的ASR模型。Whisper和fast-whisper经常产生幻觉，给出没有大写字母的长文本
- [x] 评估是否需要RAG系统来检索词汇
  - [x] 纯字符串匹配算法似乎比RAG效果更好
  - [x] 也许可以先使用字符串匹配，然后用LLM去除语义不相关的词汇？
- [x] 改进提示技术以减少token使用
  - [x] 重新格式化系统和用户提示
  - [x] 利用API提供商的提示缓存
- [ ] 添加对vllm或Llama.cpp的支持，因为它们允许运行比Ollama更多的模型（低优先级）

## 安装

假设你有一个NVIDIA GPU并且已经安装了NVIDIA驱动。如果没有，你仍然可以使用API来完成本项目中的大多数任务。我在Windows子系统Linux下的Ubuntu上运行。你的设置可能会有所不同。

你可以使用几乎任何现代版本的Python，已在`python 3.10.12`和`python 3.12.4`上测试过。
```bash
sudo apt update -y 
sudo apt upgrade -y 
sudo apt insall ffmpeg

conda create -n gat python=3.10.12
pip install -r requirements.txt
conda install cudnn # 你可能需要这个来运行faster-whisper
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

在开发这个项目时，我发现了一些提供良好UI用于翻译字幕的项目。
- [RSS-Translator](https://github.com/rss-translator/RSS-Translator)
- [video-subtitle-master](video-subtitle-master)

## 吐槽

- 这个项目起源于我想翻译一些视频到中文。现在，我希望将其构建成一个真正的工具箱。
- DeepSeek API非常便宜。它也比我能在笔记本电脑上运行的任何模型都要好。我开始认为我应该直接使用API而不是Ollama。
