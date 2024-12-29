from openai import OpenAI
from datetime import timedelta
from pydub import AudioSegment
from faster_whisper import WhisperModel
import datetime
import os
import pandas as pd


def get_whisper_prompt(dir: str) -> str:
    """Get the whisper prompt from a directory of CSV files.
    Each csv file should have a column named 'English' with the terms to be included in the prompt.

    Args:
        dir (str): The directory containing CSV files.

    Returns:
        str: comma-separated list of terms.
    """

    whisper_prompt = ""
    for file in os.listdir(dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(dir, file))
            for _, row in df.iterrows():
                whisper_prompt += row['English'] + ", "

    return whisper_prompt


def transcribe_whisper(file_name: str, model_size: str = "large-v3", whisper_prompt: str = "You are hosting a video. Please start.") -> tuple[list[datetime.timedelta], list[datetime.timedelta], list[float]]:
    """Use Faster-Whisper to transcribe a video file. The default values will run with less than 8GB of VRAM.

    Args:
        file_name (str): The name of the video file.
        model_size (str, optional): Model size. See available options on https://huggingface.co/Systran. Defaults to "large-v3".
        whisper_prompt (str, optional): An initial prompt to guide the model transcription. Defaults to "".

    Returns:
        tuple[list, list, list]: start times, end times, and text of the transcription.
    """

    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, info = model.transcribe(
        file_name, beam_size=5, initial_prompt=whisper_prompt, word_timestamps=True)

    words = []
    for segment in segments:
        words.extend(segment.words)

    start = []
    end = []
    text = []

    queue_start = None
    queue_text = ""

    for word in words:
        if queue_start is None:
            queue_start = word.start
            queue_text = word.word
        elif word.word[-1] in ".?!":
            queue_text += word.word
            start.append(datetime.timedelta(milliseconds=queue_start*1000))
            end.append(datetime.timedelta(milliseconds=word.end*1000))
            text.append(queue_text.strip(" "))
            queue_start = None
            queue_text = ""
        else:
            queue_text += word.word
    return start, end, text


def transcribe_openai(
    file_name: str,
    model: str = "whisper-1",
    api_key_env_var: str = "OPENAI_API_KEY",
    chunk_duration: int = 30000,
    timestamp_granularity: str = "segment"
) -> tuple[list[timedelta], list[timedelta], list[str]]:

    client = OpenAI(api_key=os.environ.get(api_key_env_var))

    # Load the audio file
    audio = AudioSegment.from_file(file_name)

    # Split audio into chunks
    audio_chunks = [audio[i:i + chunk_duration]
                    for i in range(0, len(audio), chunk_duration)]

    start = []
    end = []
    text = []

    for idx, chunk in enumerate(audio_chunks):
        # Export chunk to a temporary file
        chunk_file = f"chunk_{idx}.mp3"
        chunk.export(chunk_file, format="mp3")

        # Open the chunk file for transcription
        with open(chunk_file, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=audio_file,
                model=model,
                response_format="verbose_json",
                timestamp_granularities=[timestamp_granularity]
            )

        # Parse transcription results
        for segment in response.segments:
            start.append(timedelta(seconds=segment.start))
            end.append(timedelta(seconds=segment.end))
            text.append(segment.text.strip())

        # Clean up temporary file
        os.remove(chunk_file)

    return start, end, text
