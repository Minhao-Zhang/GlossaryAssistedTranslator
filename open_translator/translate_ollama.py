from collections import deque
import os
import pandas as pd
from tqdm import tqdm
import ollama
from ollama import Client

from open_translator.glossary_rag import GlossaryRAG


def translate_subtitle_v2(
    original: list[str],
    rag: GlossaryRAG = None,
    base_url: str = "http://localhost:11434",
    # this is here only to make the arguments match the OpenAI version
    api_key_env_var: str = "OLLAMA_API_KEY",
    model: str = "qwen2.5:7b",
    subtitle_history_length: int = 5
) -> list[str]:
    """Translate a list of subtitle lines using Ollama's API.

    Args:
        original (list[str]): List of original subtitle lines to translate
        rag (RAG, optional): RAG instance for glossary lookup. Defaults to None.
        base_url (str, optional): Base URL for Ollama API. Defaults to "http://localhost:11434".
        api_key_env_var (str, optional): Environment variable name containing API key. Defaults to "OLLAMA_API_KEY".
        model (str, optional): Ollama model to use. Defaults to "qwen2.5:7b".
        subtitle_history_length (int, optional): Number of previous subtitle pairs to maintain as context. Defaults to 5.

    Returns:
        list[str]: List of translated subtitle lines
    """

    # Create Ollama client with the provided base_url
    client = Client(host=base_url)

    translated = []
    history = deque(maxlen=subtitle_history_length * 2)

    with open("prompt_template/glossary_v2.txt", "r") as file:
        system_prompt = file.read()

    for i in tqdm(range(len(original))):
        if rag:
            # Query glossary definitions and format results
            glossary_df = rag.query(
                original[i],
                n_results=5
            )

            glossary = ""
            for _, row in glossary_df.iterrows():
                # glossary += f"Term: {row['Term']}\n"
                glossary += f"Definition: {row['Definition']}\n"
                glossary += f"Example: {row['Example']}\n\n"

            formmated_system_prompt = system_prompt.format(
                glossary=glossary.strip()
            )
        else:
            formmated_system_prompt = system_prompt

        # Prepare the messages for the API
        messages = [
            {
                "role": "system",
                "content": formmated_system_prompt,
            },
        ]
        messages.extend(list(history))
        messages.append({
            "role": "user",
            "content": original[i],
        })

        # Call the Ollama Chat API using the client
        response = client.chat(
            model=model,
            messages=messages
        )

        translated_text = response['message']['content'].strip()
        translated.append(translated_text)

        # Update the history
        history.append({
            "role": "user",
            "content": original[i],
        })
        history.append({
            "role": "assistant",
            "content": translated_text,
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
