"""
create_gif.py — Extract a short clip from the result video and save it as a GIF.

Usage:
    1. Open your result video in VLC and find a moment where the bounding box
       is clearly visible.  Note the timestamp (e.g. 0:42).
    2. Set START_SECONDS and DURATION_SECONDS below.
    3. Run:  python create_gif.py
    4. Commit demo.gif and add it to the README.
"""

import cv2
from PIL import Image
from pathlib import Path

# ── Configure these two values ────────────────────────────────────────────────
START_SECONDS   = 559    # second in the video to start the GIF
DURATION_SECONDS = 6    # how many seconds to capture
# ─────────────────────────────────────────────────────────────────────────────

INPUT_VIDEO  = Path("nuggets_thunder_05172026_01.mp4")
OUTPUT_GIF   = Path("demo.gif")
GIF_FPS      = 8     # lower = smaller file size; 8 looks smooth enough for a demo
GIF_WIDTH    = 640   # resize width in pixels; height scales automatically


def create_gif(video_path: Path, output_path: Path,
               start: float, duration: float,
               fps: int, width: int) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    source_fps   = cap.get(cv2.CAP_PROP_FPS)
    start_frame  = int(start * source_fps)
    end_frame    = int((start + duration) * source_fps)
    # Sample every Nth source frame so the GIF plays at GIF_FPS
    frame_step   = max(1, round(source_fps / fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    frame_num = start_frame
    while frame_num < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_num - start_frame) % frame_step == 0:
            # OpenCV uses BGR; Pillow expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            # Resize while preserving aspect ratio
            h = int(img.height * width / img.width)
            img = img.resize((width, h), Image.LANCZOS)
            frames.append(img)
        frame_num += 1

    cap.release()

    if not frames:
        print("No frames captured — check START_SECONDS and the video length.")
        return

    frame_duration_ms = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,                      # 0 = loop forever
        duration=frame_duration_ms,
        optimize=True,
    )

    size_mb = output_path.stat().st_size / 1_000_000
    print(f"Saved {output_path}  ({len(frames)} frames, {size_mb:.1f} MB)")
    if size_mb > 10:
        print("Tip: reduce DURATION_SECONDS or GIF_WIDTH to shrink the file.")


if __name__ == "__main__":
    create_gif(INPUT_VIDEO, OUTPUT_GIF, START_SECONDS, DURATION_SECONDS, GIF_FPS, GIF_WIDTH)
