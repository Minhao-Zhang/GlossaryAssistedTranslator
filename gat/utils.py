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
            raise FileNotFoundError(
                f"Cookie file not found: {use_cookie_file}")
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
            raise FileNotFoundError(
                f"Cookie file not found: {use_cookie_file}")
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


def save_ass(file_name: str, start: list, end: list, text: list, davinci: bool = False) -> None:
    """
    Save subtitles in ASS format with a gray transparent shadow background, 
    black outline, bold font, and proper drop shadow.

    Args:
        file_name (str): The name of the file to save the subtitles.
        start (list): List of start times as datetime.timedelta objects.
        end (list): List of end times as datetime.timedelta objects.
        text (list): List of subtitle texts.
        davinci (bool): Adjust times for DaVinci Resolve timeline if True.
    """
    # Define a list of 4 soft but visible colors for speakers (Blue for Speaker 0, Orange for Speaker 1, etc.)
    colors = [
        "&H0099CCFF",  # Soft Blue (Speaker 0)
        "&H00FFCC99",  # Soft Orange (Speaker 1)
        "&H00A3FFA3",  # Soft Green (Speaker 2)
        "&H00E6E6E6"   # Light Gray (Speaker 3)
    ]

    # Map speakers to colors (limit to 4)
    speaker_colors = {}
    next_color_index = 0

    with open(file_name, 'w', encoding='utf-8') as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 640\n")
        f.write("PlayResY: 480\n")
        f.write("\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write("Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,3,2,10,10,10,0\n")
        f.write("\n")
        f.write("[Events]\n")
        f.write(
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        for i, (s, e, t) in enumerate(zip(start, end, text)):
            # Extract speaker from the text and remove it
            speaker = t.split(':')[0].strip('[]')
            subtitle_text = ':'.join(t.split(':')[1:]).strip()

            if speaker not in speaker_colors:
                if len(speaker_colors) < 4:
                    speaker_colors[speaker] = colors[next_color_index %
                                                     len(colors)]
                    next_color_index += 1
                else:
                    # Cycle through 4 colors
                    speaker_colors[speaker] = colors[next_color_index % 4]
                    next_color_index += 1

            # Calculate total seconds and centiseconds
            ss = int(s.total_seconds())
            es = int(e.total_seconds())
            sms = s.microseconds // 1000
            ems = e.microseconds // 1000

            # Adjust for DaVinci Resolve timeline
            if davinci:
                ss += 3600
                es += 3600

            # Convert to centiseconds (0-99)
            sms_centi = sms // 10
            ems_centi = ems // 10

            # Format start and end times
            start_time = f"{ss//3600:01}:{(ss % 3600)//60:02}:{ss % 60:02}.{sms_centi:02}"
            end_time = f"{es//3600:01}:{(es % 3600)//60:02}:{es % 60:02}.{ems_centi:02}"

            # Write the subtitle block to the file with selected color
            f.write(
                f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{{\b1\3c&H000000&\4c&H80000000&\shad3\c{speaker_colors[speaker]}}}{subtitle_text}\n")
