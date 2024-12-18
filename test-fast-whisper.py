from faster_whisper import WhisperModel
import datetime

filename = "cache/W_eaNIUDOq8.webm"
model_size = "large-v3" # uses less than 6GB VRAM

import pandas as pd 

agents = pd.read_csv("valorant/agents.csv")
agents2 = pd.read_csv("valorant/agents2.csv")
games = pd.read_csv("valorant/games.csv")
guns = pd.read_csv("valorant/guns.csv")
maps = pd.read_csv("valorant/maps.csv")
teams = pd.read_csv("valorant/teams.csv")

whisper_prompt = "Glossary: "
for _, row in agents.iterrows():
    whisper_prompt += row['English'] + ", "
for _, row in guns.iterrows():
    whisper_prompt += row['English'] + ", "
for _, row in games.iterrows():
    whisper_prompt += row['English'] + ", "
for _, row in teams.iterrows():
    whisper_prompt += row['English'] + ", "

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe(filename, beam_size=5, initial_prompt=whisper_prompt)
    

def extract_transcription(segments) -> tuple[list, list, list]:
    """Extract the transcription into start times, end times, and text.

    Args:
        transcription (dict): The transcription result from whisper.

    Returns:
        tuple[list, list, list]: A tuple containing the start times, end times, and text.
    """

    start = []
    end = []
    original = []
    for segment in segments:
        start.append(datetime.timedelta(milliseconds=segment.start*1000))
        end.append(datetime.timedelta(milliseconds=segment.end*1000))
        original.append(segment.text.strip())

    return start, end, original

start, end, english = extract_transcription(segments)

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

save_srt(f"{filename}.en.srt", start, end, english)