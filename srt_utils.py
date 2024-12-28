
import datetime


def save_srt(file_name: str, start: list, end: list, text: list, davinci: bool = False) -> None:
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


def save_bilingual_srt(file_name: str, start: list, end: list, text: list, translation: list, davinci: bool = False) -> None:
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
