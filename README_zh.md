# 术语辅助翻译器

[English](README.md) | 中文版

一个支持术语表辅助翻译的通用工具箱

- 针对字幕翻译优化
- 支持本地模型或API调用
- 通过术语表实现低资源消耗翻译
- （计划中）自动字幕分段与合并功能

您只需准备好术语表文件即可开启翻译之旅

## 安装部署

本项目具有高度灵活性，支持从全本地运行到完全API调用的多种部署方案

### 全本地推理部署方案

假设您已配备NVIDIA显卡并安装驱动程序。若无GPU仍可通过API完成大部分功能。测试环境为Windows子系统Ubuntu，不同系统配置可能略有差异。

支持Python 3.10+版本，已在`python 3.10.12`和`python 3.12.4`验证。主要受限于依赖项`faster-whisper`。

```bash
sudo apt update -y 
sudo apt upgrade -y 
sudo apt install ffmpeg

conda create -n gat python=3.10.12
pip install -r requirements.txt
conda install cudnn # 运行faster-whisper可能需要此依赖
```

此环境配置支持本地运行语音识别模型及调用各类大语言模型API。若需完全本地化翻译，建议安装Ollama并下载所需LLM模型。qwen2.5系列模型因其多语言支持和灵活尺寸可作为初始选择，实际应用中可根据需求测试不同模型效果。

## 术语表格式规范

### 术语匹配表格式

请将所有术语表文件置于指定目录（默认为`data`），支持单文件整合或多文件分列管理。

格式要求：

- 必须使用CSV格式
- 须包含四个核心字段：Term（术语）、Translation（译词）、Definition（定义）、Example（用例）
- 
其他字段可作为元数据自由添加

示例文件：

```csv
"Term","Translation","Definition","Example"
"Ollama","Ollama","一键式本地运行大语言模型的软件","I prefer Ollama over vllm because it is simple. --> 相比vllm，我还是更喜欢Ollama的简洁。"
"Whisper","Whisper","OpenAI开发的语音识别模型","Whisper is a ASR model develoepd by openAI. --> Whisper是一个由openAI开发的自动语言识别模型。"
```

## 术语校正表格式

采用JSON格式，键名为标准术语，键值为常见变体列表。

示例文件：

```json
{
    "Ollama": ["Ohllama"],
    "Qwen": ["Quwen", "Kuen"]
}
```

## 快速入门

请参考示例文件example.ipynb

## 说话人识别字幕

基于whisperX实现说话人识别功能，可在转写字幕前自动添加[SPEAKER_00]: 标识（数字代表不同说话人）。

注：因环境配置问题，Python直接集成whisperX未获成功。推荐使用[jim60105制作的docker镜像](https://github.com/jim60105/docker-whisperX)。

在`scripts/`目录中提供了Windows PowerShell和Bash脚本范例，供参考使用。

## 同类项目

开发过程中发现以下优秀的字幕翻译工具：

- [RSS-Translator](https://github.com/rss-translator/RSS-Translator)
- [video-subtitle-master](video-subtitle-master)

## 开发手记

- 深度求索API的价格非常低廉，效果远超本地模型，正考虑全面转向API方案
- 深切哀悼被DDOS攻击的深度求索服务器
