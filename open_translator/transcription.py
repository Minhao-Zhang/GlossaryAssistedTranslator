"""
Audio transcription module supporting multiple transcription backends.

This module provides functionality for transcribing audio files using either
Faster-Whisper or OpenAI's Whisper API. It includes support for handling large files
through chunking and various transcription options.
"""

# Standard library imports
import os
import datetime
from datetime import timedelta

# Third-party imports
import pandas as pd
from pydub import AudioSegment
from openai import OpenAI
from faster_whisper import WhisperModel
from typing import List, Tuple


def get_whisper_prompt(dir: str) -> str:
    """
    Build a whisper prompt from CSV files containing special terms.

    Scans a directory for CSV files containing specialized vocabulary or terms
    that should be included in the transcription prompt. Each CSV must have an
    'English' column.

    Args:
        dir: Directory path containing CSV files with vocabulary terms.

    Returns:
        A comma-separated string of terms to use as a prompt.
    """

    whisper_prompt = ""
    for file in os.listdir(dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(dir, file))
            for _, row in df.iterrows():
                whisper_prompt += row['English'] + ", "

    return whisper_prompt


def transcribe_whisper(
    file_name: str,
    model_size: str = "large-v3",
    whisper_prompt: str = "You are hosting a video. Please start."
) -> Tuple[List[timedelta], List[timedelta], List[str]]:
    """
    Transcribe audio using Faster-Whisper model with GPU acceleration.

    Performs word-level transcription with automatic sentence segmentation.
    Optimized for CUDA devices with limited VRAM (< 8GB).
    See which models to use at https://github.com/SYSTRAN/faster-whisper and https://huggingface.co/Systran.

    Args:
        file_name: Path to the audio/video file.
        model_size: Whisper model size (tiny, base, small, medium, large-v1, large-v2, large-v3).
        whisper_prompt: Initial prompt to guide the transcription context.

    Returns:
        Tuple containing:
        - List of sentence start times
        - List of sentence end times
        - List of transcribed text segments
    """

    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, _ = model.transcribe(
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
) -> Tuple[List[timedelta], List[timedelta], List[str]]:
    """
    Transcribe audio using OpenAI's Whisper API with automatic chunking.

    Handles large files by splitting them into smaller chunks before transcription.
    Supports various timestamp granularities for precise timing information.

    Args:
        file_name: Path to the audio/video file.
        model: OpenAI Whisper model identifier.
        api_key_env_var: Environment variable name containing the OpenAI API key.
        chunk_duration: Duration of each audio chunk in milliseconds.
        timestamp_granularity: Granularity of timestamp information ('word' or 'segment').

    Returns:
        Tuple containing:
        - List of segment start times
        - List of segment end times
        - List of transcribed text segments

    Raises:
        EnvironmentError: If the API key environment variable is not set.
    """

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
