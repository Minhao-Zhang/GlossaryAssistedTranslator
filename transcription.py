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


def transcribe(file_name: str, model_size: str = "large-v3", whisper_prompt: str = "") -> tuple[list[datetime.timedelta], list[datetime.timedelta], list[float]]:
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
        file_name, beam_size=5, initial_prompt=whisper_prompt)

    start = []
    end = []
    text = []

    for segment in segments:
        # convert start into timedelta
        start.append(datetime.timedelta(milliseconds=segment.start*1000))
        end.append(datetime.timedelta(milliseconds=segment.end*1000))
        text.append(segment.text.strip())

    return start, end, text
