# TODO: Test

import yt_dlp
import re

def get_top_videos_for_level(level_name, author_name, level_id, **kwargs):
    # Query format used by gdbrowser.com
    query = f"Geometry Dash {level_name} by {author_name} {str(level_id)}"

    get_top_videos(query, **kwargs)

def get_top_videos(query: str, num_results: int = 1, output_dir: str = ".", download_delay: int = 2):
    query_file_name = re.sub(r'[^a-zA-Z0-9 ]', '', query).replace(" ", "_")

    download_options = {
        # Want a height of 720 (or lower if no other options. Audio doesn't matter.
        "format": "bestvideo[height=720][ext=mp4]/bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]",
        "outtmpl": f"{output_dir}/{query_file_name}.%(ext)s",
        "restrictfilenames": True,
        "noplaylist": True,
        "sleep_interval": download_delay
    }

    with yt_dlp.YoutubeDL(download_options) as ydl:
        ydl.download([f"ytsearch{num_results}:{query}"])