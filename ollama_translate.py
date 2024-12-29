from collections import deque
import json
import os
import pandas as pd
from tqdm import tqdm
import ollama

from rag import RAG


def translate_subtitle_v2(original: list[str], rag: RAG, model: str = "qwen2.5:7b", subtitle_history_length: int = 5) -> list[str]:
    # necessary for docker to communicate with host
    # client = Client(host='http://host.docker.internal:11434')

    translated = []
    history = deque(maxlen=subtitle_history_length*2)

    for i in tqdm(range(len(original))):

        docs = rag.query(
            "Find possible definition involved in words in the sentence below: " + original[i], n_results=5)
        glossary = ""
        for doc in docs:
            glossary += "Definition: " + doc.page_content + \
                "\nSample Translation: " + \
                doc.metadata["example_translation"] + "\n"

        system_prompt = f"""
        You are a multilingual subtitle translation assistant. Translate the following English subtitle into Simplified Chinese (Mandarin).
        This is a Valorant analysis video, and you may encounter technical terms related to the game.
        The word "Vitality" should always be translated to "VIT" and "Trace" or "Trace Esports" should always be translated to "TE".
        Refer to the following definition of technical terms and example translation:
        ---
        {glossary}
        ---
        Ensure the translation is natural and culturally localized, avoiding overly direct English phrasing.
        Consider the preceding and following sentences to maintain contextual accuracy and flow.
        The word "Vitality" should always be translated to "VIT" and "Trace" or "Trace Esports" should always be translated to "TE".

        Reply **only** with the translated text and nothing else.
        """

        # prepare the history
        messages = [
            {
                'role': 'system',
                'content': system_prompt,
            },
        ]
        messages.extend(list(history))
        messages.append({
            'role': 'user',
            'content': "Translate the following sentence while pay attention to the sentence before this: " + original[i],
        })

        # chat with the model
        response = ollama.chat(model=model, messages=messages)
        translated.append(response['message']['content'].strip())

        # update the history
        history.append({
            'role': 'user',
            'content': "Translate the following sentence while pay attention to the sentence before this: " + original[i],
        })
        history.append({
            'role': 'assistant',
            'content': translated[-1],
        })

    unload_model(model)
    return translated


def translate_subtitle_v1(original: list[str], system_prompt: str, model: str = "qwen2.5:7b", subtitle_history_length: int = 5, all_glossary: str = "") -> list[str]:
    """
    Translate a list of original_language sentences to another language by setting LLM prompt.

    Args:
        original (list[str]): A list of sentences to be translated.
        system_prompt (str): The system prompt to use for translation.
        model (str, optional): The Ollama model to use for translation. Defaults to "qwen2.5:7b".
        subtitle_history_length (int, optional): The length of the history to maintain. Defaults to 5.
        all_glossary (str): The glossary to use for translation.

    Returns:
        list[str]: A list of translations.
    """

    # necessary for docker to communicate with host
    # client = Client(host='http://host.docker.internal:11434')

    translated = []
    history = deque(maxlen=subtitle_history_length*2)

    for i in tqdm(range(len(original))):
        messages = [
            {
                'role': 'system',
                'content': f"""
                You are an assisstant for an English-Mandarin professional interpreter.
                You are tasked to find possible glossary terms in the conversation.

                Provide the glossary terms in the following format:
                Original: Translation, Original: Translation, ...
                Here is the source glossary:
                ---
                {all_glossary}
                ---
                Only provide the glossary terms and their coresponding translation that are mentioned in the content.
                If none of the source glossary terms are mentioned in the content, please respond "No glossary".
                """,
            },
            {
                'role': 'user',
                'content': f"{original[i]}"
            }
        ]

        response = ollama.chat(model=model, messages=messages)
        concise_glossary = response['message']['content'].strip()

        system_prompt = f"""
        You are a multilingual subtitle translation assistant. Translate the following English subtitle into Simplified Chinese (Mandarin).
        This is a Valorant analysis video, and you may encounter technical terms related to the game.
        Refer to the following glossary for key terminology, but adapt as needed for any transcription errors or approximations:
        Here is the glossary with format English: 中文
        ---
        {concise_glossary}
        ---

        Ensure the translation is natural and culturally localized, avoiding overly direct English phrasing.
        Consider the preceding and following sentences to maintain contextual accuracy and flow.

        Reply **only** with the translated text and nothing else.
        """

        # prepare the history
        messages = [
            {
                'role': 'system',
                'content': system_prompt,
            },
        ]
        messages.extend(list(history))
        messages.append({
            'role': 'user',
            'content': original[i],
        })

        # chat with the model
        response = ollama.chat(model=model, messages=messages)
        translated.append(response['message']['content'].strip())

        # update the history
        history.append({
            'role': 'user',
            'content': original[i],
        })
        history.append({
            'role': 'assistant',
            'content': translated[-1],
        })

    unload_model(model)
    return translated


def unload_model(model: str):
    # client = Client(host='http://host.docker.internal:11434')
    response = ollama.generate(model=model, keep_alive=0)
    return response


def get_glossary(dir: str) -> str:
    # get all csv files
    csv_files = [f for f in os.listdir(dir) if f.endswith('.csv')]
    json_files = [f for f in os.listdir(dir) if f.endswith('.json')]

    glossary = ""

    for file in csv_files:
        df = pd.read_csv(os.path.join(dir, file))
        for _, row in df.iterrows():
            glossary += row['English'] + ": " + row['Chinese'] + ", "

    # for file in json_files:
    #     with open(os.path.join(dir, file), 'r') as f:
    #         data = json.load(f)

    #     for english_key, details in data.items():
    #         # Ensure there is a "Chinese" key in the details
    #         if "Chinese" in details:
    #             # Add the top-level English-Chinese pair
    #             glossary += english_key + ": " + details['Chinese'] + ", "

    #         # Extract all other English-Chinese pairs within this section
    #         for sub_key, sub_value in details.items():
    #             # Skip the "Chinese" key itself
    #             if sub_key != "Chinese":
    #                 glossary += sub_key + ": " + sub_value + ", "

    return glossary

# A simple rule of thumb for the model size is
# 140 words per minute in normal English speech and 4 tokens for 3 words
# So for a 10-minute video, you will need a context length about 2k.
# There's definitely a lot of variance in this, but it's a good starting point
# You also need to add in overhead like system prompts and user messages
# Since Ollama defaults to 2k context for all models, you probably need a larger model
# You can use the provided model file to set the context length to 4k or 8k


def summarize_transcription(transcription: dict, model: str = "8k-qwen2.5:7b") -> str:
    """Summarize the transcription into a concise format using LLM.

    Args:
        transcription (dict): The transcription result from whisper.
        model (str, optional): The Ollama model to use for summarization. Defaults to "8k-qwen2.5:7b".

    Returns:
        str: The summarized transcription.
    """

    # necessary for docker to communicate with host
    # client = Client(host='http://host.docker.internal:11434')

    system_prompt = """
    Summarize the following text into a concise and coherent paragraph.
    The text is a transcription of a video.
    Ensure the summary captures the main points of the video.
    Keep the summary to a maximum of 50 words.
    Only reply with the summary text, nothing else.
    Here is the transcription:
    """

    response = ollama.chat(model=model, messages=[
        {
            'role': 'system',
            'content': system_prompt,
        },
        {
            'role': 'user',
            'content': transcription['text'],
        },
    ])

    unload_model(model)

    return response['message']['content'].strip()
