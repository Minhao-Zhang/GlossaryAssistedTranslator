from collections import deque
from tqdm import tqdm
import ollama

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


def translate_subtitle(original: list[str], system_prompt: str, model: str = "qwen2.5:7b", subtitle_history_length: int = 10) -> list[str]:
    """
    Translate a list of original_language sentences to another language by setting LLM prompt.

    Args:
        original (list[str]): A list of sentences to be translated.
        system_prompt (str): The system prompt to use for translation.
        model (str, optional): The Ollama model to use for translation. Defaults to "qwen2.5:7b".

    Returns:
        list[str]: A list of translations.
    """

    # necessary for docker to communicate with host
    # client = Client(host='http://host.docker.internal:11434')

    translated = []
    history = deque(maxlen=subtitle_history_length*2)

    for i in tqdm(range(len(original))):

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

        # print(f"Original   : {original[i]}")
        # print(f"Translation: {translated[-1]}")

    unload_model(model)
    return translated


def unload_model(model: str):
    """Unload Ollama model.

    Args:
        model (str): The Ollama model to unload.

    Returns:
        any: The response from the Ollama API.
    """
    # client = Client(host='http://host.docker.internal:11434')
    response = ollama.generate(model=model, keep_alive=0)
    return response
