from faster_whisper import WhisperModel
import datetime


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
