"""
SRT Utilities Module

This module provides utility functions for handling SRT (SubRip Subtitle) files.
It includes functions to save subtitles in SRT format, load subtitles from an SRT file,
and save bilingual subtitles in SRT format. The functions support precise timing and
optional adjustments for DaVinci Resolve timelines.
"""

import datetime


def save_srt(file_name: str, start: list, end: list, text: list, davinci: bool = False) -> None:
    """
    Save subtitles in SRT format.

    Args:
        file_name (str): The name of the file to save the subtitles.
        start (list): List of start times as datetime.timedelta objects.
        end (list): List of end times as datetime.timedelta objects.
        text (list): List of subtitle texts.
        davinci (bool): Adjust times for DaVinci Resolve timeline if True.
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


def load_srt(file_name: str) -> tuple[list, list, list]:
    """
    Load subtitles from an SRT file.

    Args:
        file_name (str): The name of the SRT file to load.

    Returns:
        tuple: Three lists containing start times, end times, and texts.
    """
    start = []
    end = []
    text = []

    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        try:
            # Skip empty lines
            if lines[i].strip() == '':
                i += 1
                continue

            # Read subtitle index (e.g., "1", "2", ...)
            subtitle_index = int(lines[i].strip())
            i += 1

            # Extract start and end times
            start_time, end_time = lines[i].strip().split(' --> ')
            start_dt = datetime.datetime.strptime(start_time, '%H:%M:%S,%f')
            end_dt = datetime.datetime.strptime(end_time, '%H:%M:%S,%f')

            # Convert datetime to timedelta
            start.append(datetime.timedelta(
                hours=start_dt.hour, minutes=start_dt.minute, seconds=start_dt.second, milliseconds=start_dt.microsecond // 1000))
            end.append(datetime.timedelta(
                hours=end_dt.hour, minutes=end_dt.minute, seconds=end_dt.second, milliseconds=end_dt.microsecond // 1000))
            i += 1

            # Extract subtitle text (handle multi-line text)
            subtitle_text = []
            while i < len(lines) and lines[i].strip() != '':
                subtitle_text.append(lines[i].strip())
                i += 1
            text.append('\n'.join(subtitle_text))

        except (IndexError, ValueError) as e:
            print(f"Error processing SRT block starting at line {i + 1}: {e}")
            i += 1

    return start, end, text


def save_bilingual_srt(file_name: str, start: list, end: list, text: list, translation: list, davinci: bool = False) -> None:
    """
    Save bilingual subtitles in SRT format.

    Args:
        file_name (str): The name of the file to save the subtitles.
        start (list): List of start times as datetime.timedelta objects.
        end (list): List of end times as datetime.timedelta objects.
        text (list): List of original subtitle texts.
        translation (list): List of translated subtitle texts.
        davinci (bool): Adjust times for DaVinci Resolve timeline if True.
    """
    with open(file_name, 'w', encoding='utf-8') as f:
        for i, (s, e, t, tr) in enumerate(zip(start, end, text, translation)):
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
            f.write(f"{tr}\n\n")
