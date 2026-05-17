"""
1_download_video.py — Step 1: Download the source game footage from YouTube.

Uses yt-dlp to fetch the best available video quality and automatically
convert it to MP4. The output file is named after the video's title.

Usage:
    python 1_download_video.py

Requirements:
    pip install yt-dlp
"""

from yt_dlp import YoutubeDL
from config import YOUTUBE_URL


def download_video(url: str) -> None:
    """
    Download a YouTube video as an MP4 file to the current directory.

    Args:
        url: Full YouTube watch URL (e.g. "https://www.youtube.com/watch?v=...").
    """
    options = {
        # Select the highest-quality video stream available.
        "format": "bv*",

        # Name the output file after the video's title (YouTube provides this).
        "outtmpl": "%(title)s.%(ext)s",

        # After download, convert the container to MP4 so every other step
        # in this pipeline can open it with OpenCV without extra configuration.
        "postprocessors": [{
            "key": "FFmpegVideoConvertor",
            "preferedformat": "mp4",
        }],
    }

    print(f"Downloading: {url}")
    with YoutubeDL(options) as ydl:
        ydl.download([url])
    print("Download complete.")


if __name__ == "__main__":
    download_video(YOUTUBE_URL)
