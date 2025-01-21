import datetime
import os

import yt_dlp


def download_video(url: str, output_dir: str = 'cache', use_cookie_file=None) -> str:
    """Download video from a YouTube video or other source.

    Args:
        url (str): The URL of the video.
        output_dir (str, optional): Output directory. Defaults to 'cache'.
        use_cookie_file (str, optional): Path to cookie file for authentication. Defaults to None.

    Returns:
        str: The path to the downloaded video file.

    Raises:
        FileNotFoundError: If cookie file is specified but does not exist
    """

    # check the output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if use_cookie_file:
        if not os.path.exists(use_cookie_file):
            raise FileNotFoundError(f"Cookie file not found: {use_cookie_file}")
        ydl_opts = {'format': "bestvideo+bestaudio",
                    'cookiefile': use_cookie_file,
                    'outtmpl': f'{output_dir}/%(id)s.%(ext)s'}
    else:
        ydl_opts = {'format': "bestvideo+bestaudio",
                    'outtmpl': f'{output_dir}/%(id)s.%(ext)s'}
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url)

    return f'{output_dir}/{info["id"]}.{info["ext"]}'


def download_audio(url: str, output_dir: str = 'cache', use_cookie_file=None) -> str:
    """Download video from a YouTube audio or other source.

    Args:
        url (str): The URL of the audio.
        output_dir (str, optional): Output directory. Defaults to 'cache'.
        use_cookie_file (str, optional): Path to cookie file for authentication. Defaults to None.

    Returns:
        str: The path to the downloaded audio file.

    Raises:
        FileNotFoundError: If cookie file is specified but does not exist
    """

    # check the output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if use_cookie_file:
        if not os.path.exists(use_cookie_file):
            raise FileNotFoundError(f"Cookie file not found: {use_cookie_file}")
        ydl_opts = {'format': "bestaudio",
                    'cookiefile': use_cookie_file,
                    'outtmpl': f'{output_dir}/%(id)s.%(ext)s'}
    else:
        ydl_opts = {'format': "bestaudio",
                    'outtmpl': f'{output_dir}/%(id)s.%(ext)s'}
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url)

    return f'{output_dir}/{info["id"]}.{info["ext"]}'


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


def split_subtitles(start: list[datetime.timedelta], end: list[datetime.timedelta], sentences: list[str]) -> tuple[list, list, list]:
    """
    Split Chinese subtitle sentences if they exceed 40 characters.

    Args:
        start (list): List of start times as datetime.timedelta objects
        end (list): List of end times as datetime.timedelta objects
        sentences (list): List of Chinese sentences to potentially split

    Returns:
        tuple: Three lists containing new start times, end times, and texts
               with split sentences and adjusted timings
    """
    new_start = []
    new_end = []
    new_sentences = []

    for start_time, end_time, sentence in zip(start, end, sentences):
        if len(sentence) <= 40:
            new_start.append(start_time)
            new_end.append(end_time)
            new_sentences.append(sentence)
            continue

        # Find all Chinese commas in the sentence
        commas = [i for i, char in enumerate(sentence) if char == '，']
        
        if not commas:
            new_start.append(start_time)
            new_end.append(end_time)
            new_sentences.append(sentence)
            continue

        # Find the comma closest to the middle
        middle = len(sentence) // 2
        split_index = min(commas, key=lambda x: abs(x - middle)) + 1  # Include the comma

        # Split the sentence
        part1 = sentence[:split_index]
        part2 = sentence[split_index:]

        # Calculate time proportions
        total_length = len(sentence)
        part1_ratio = len(part1) / total_length
        duration = (end_time - start_time).total_seconds()

        # Calculate new end time for part1 with 50ms gap
        part1_end = start_time + datetime.timedelta(seconds=duration * part1_ratio) - datetime.timedelta(milliseconds=50)
        
        # Calculate new start time for part2 with 50ms gap
        part2_start = part1_end + datetime.timedelta(milliseconds=100)

        # Ensure the gap doesn't make the duration negative
        if part1_end > start_time and part2_start < end_time:
            new_start.append(start_time)
            new_end.append(part1_end)
            new_sentences.append(part1)

            new_start.append(part2_start)
            new_end.append(end_time)
            new_sentences.append(part2)
        else:
            # If gap would make duration negative, don't split
            new_start.append(start_time)
            new_end.append(end_time)
            new_sentences.append(sentence)
    
    return new_start, new_end, new_sentences
