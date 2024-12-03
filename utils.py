import os
import datetime
from tqdm import tqdm
import torch
import whisper
import yt_dlp
from ollama import Client


def download_video(url: str, output_dir: str = 'cache') -> str:
    """Download video from a YouTube video or other source.

    Args:
        url (str): The URL of the video.
        output_dir (str, optional): Output directory. Defaults to 'cache'.

    Returns:
        str: The path to the downloaded audio file.
    """

    # check the output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ydl_opts = {'format': "bestvideo+bestaudio",
                # 'cookiefile': 'ytb_cookies.txt',
                'outtmpl': f'{output_dir}/%(id)s.%(ext)s'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url)

    return f'{output_dir}/{info["id"]}.{info["ext"]}'


def transcribe_audio(filename: str, whisper_model: str = "medium", language='en') -> dict:
    """Transcribe audio from a file.

    Args:
        filename (str): The path to the audio file.
        whisper_model (str, optional): OpenAI Whisper Model. Defaults to "medium". See more on https://github.com/openai/whisper.
        language (str, optional): The language of the audio. Defaults to 'en'.
    Returns:
        dict: The transcription result.
    """

    # Load the model, use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(
        whisper_model, device=device)

    result = model.transcribe(
        filename,
        language=language,
        verbose=False
    )
    del model
    torch.cuda.empty_cache()
    return result


def extract_transcription(transcription: dict) -> tuple[list, list, list]:
    """Extract the transcription into start times, end times, and text.

    Args:
        transcription (dict): The transcription result from whisper.

    Returns:
        tuple[list, list, list]: A tuple containing the start times, end times, and text.
    """

    start = []
    end = []
    original = []
    for segment in transcription['segments']:
        start.append(datetime.timedelta(milliseconds=segment['start']*1000))
        end.append(datetime.timedelta(milliseconds=segment['end']*1000))
        original.append(segment['text'].strip())

    return start, end, original


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
    client = Client(host='http://host.docker.internal:11434')

    system_prompt = """
    Summarize the following text into a concise and coherent paragraph.
    The text is a transcription of a video.
    Ensure the summary captures the main points of the video.
    Keep the summary to a maximum of 50 words.
    Only reply with the summary text, nothing else.
    Here is the transcription:
    """

    response = client.chat(model=model, messages=[
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


def translate_transcription(original: list[str], system_prompt: str, model: str = "qwen2.5:7b") -> list[str]:
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
    client = Client(host='http://host.docker.internal:11434')

    # A sample system prompt for translation could be

    # Translate the following English text to {language}.
    # The text is part of a {video_context} video script.
    # Ensure the translation feels natural and culturally localized, avoiding direct English phrasing where possible.
    # Maintain a tone that is light, friendly, and suitable for the context, but not overly cheerful.
    # Keep the translation concise.
    # Only reply with the translation text, nothing else.

    translated = []
    for line in tqdm(original):
        response = client.chat(model=model, messages=[
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': line,
            },
        ])
        translated.append(response['message']['content'].strip())

    unload_model(model)
    return translated


def unload_model(model: str):
    """Unload Ollama model.

    Args:
        model (str): The Ollama model to unload.

    Returns:
        any: The response from the Ollama API.
    """
    client = Client(host='http://host.docker.internal:11434')
    response = client.generate(model=model, keep_alive=0)
    return response


def save_srt(file_name: str, start: list, end: list, text: list, davinci: bool = False) -> None:
    """Save the transcription to an SRT file.

    Args:
        file_name (str): The name of the output file.
        start (list): A list of start times.
        end (list): A list of end times.
        text (list): A list of text.
        davinci (bool, optional): Whether to adjust for DaVinci Resolve timeline. Defaults to False.
    """

    with open(file_name, 'w') as f:
        for i, (s, e, t) in enumerate(zip(start, end, text)):
            # Convert total seconds to integer and extract milliseconds
            sms = s.microseconds // 1000
            ems = e.microseconds // 1000

            ss = s.seconds
            es = e.seconds

            # Adjust for DaVinci Resolve timeline
            if davinci:
                ss += 3600
                es += 3600

            # Format start and end times with precise milliseconds
            s = f"{ss//3600:02}:{(ss % 3600)//60:02}:{ss % 60:02},{sms:03}"
            e = f"{es//3600:02}:{(es % 3600)//60:02}:{es % 60:02},{ems:03}"

            # Write the subtitle block to the file
            f.write(f"{i+1}\n")
            f.write(f"{s} --> {e}\n")
            f.write(f"{t}\n\n")


def save_bilingual_srt(file_name: str, start: list, end: list, translated: list, original: list, davinci: bool = False) -> None:
    """Save the transcription to an SRT file.

    Args:
        file_name (str): The name of the output file.
        start (list): A list of start times.
        end (list): A list of end times.
        text (list): A list of text.
        davinci (bool, optional): Whether to adjust for DaVinci Resolve timeline. Defaults to False.
    """

    with open(file_name, 'w') as f:
        for i, (s, e, o, t) in enumerate(zip(start, end, original, translated)):
            # Convert total seconds to integer and extract milliseconds
            sms = s.microseconds // 1000
            ems = e.microseconds // 1000

            ss = s.seconds
            es = e.seconds

            # Adjust for DaVinci Resolve timeline
            if davinci:
                ss += 3600
                es += 3600

            # Format start and end times with precise milliseconds
            s = f"{ss//3600:02}:{(ss % 3600)//60:02}:{ss % 60:02},{sms:03}"
            e = f"{es//3600:02}:{(es % 3600)//60:02}:{es % 60:02},{ems:03}"

            # Write the subtitle block to the file
            f.write(f"{i+1}\n")
            f.write(f"{s} --> {e}\n")
            f.write(f"{t}\n")
            f.write(f"{o}\n\n")


def load_srt(file_name: str) -> tuple[list, list, list]:
    """Load an SRT file into start times, end times, and text.

    Args:
        file_name (str): The path to the SRT file.

    Returns:
        tuple[list, list, list]: A tuple containing the start times (list of timedelta), 
                                 end times (list of timedelta), and text (list of strings).
    """
    start = []
    end = []
    text = []

    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Parse the SRT lines
    for i in range(0, len(lines), 4):
        try:
            # Extract start and end times
            start_time, end_time = lines[i + 1].strip().split(' --> ')

            # Extract text
            text.append(lines[i + 2].strip())

            # Convert times to timedelta
            start_dt = datetime.datetime.strptime(start_time, '%H:%M:%S,%f')
            end_dt = datetime.datetime.strptime(end_time, '%H:%M:%S,%f')

            start.append(datetime.timedelta(
                hours=start_dt.hour, minutes=start_dt.minute, seconds=start_dt.second, milliseconds=start_dt.microsecond // 1000))
            end.append(datetime.timedelta(
                hours=end_dt.hour, minutes=end_dt.minute, seconds=end_dt.second, milliseconds=end_dt.microsecond // 1000))
        except (IndexError, ValueError) as e:
            print(f"Error processing SRT block starting at line {i + 1}: {e}")

    return start, end, text
