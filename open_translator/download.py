import os
import yt_dlp


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
