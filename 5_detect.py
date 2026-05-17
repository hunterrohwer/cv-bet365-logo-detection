"""
5_detect.py — Step 5: Run logo detection on the source video and save the result as MP4.

Loads the trained model weights, processes the source video frame by frame,
draws a bounding box around every detected Bet365 logo, and writes the
annotated frames to a new MP4 file.

Each run is saved next to the source video with the date and a run counter
appended, e.g. nuggets_thunder_05172026_01.mp4.  The counter increments
automatically so no run ever overwrites a previous one.

Usage:
    python 5_detect.py

Requirements:
    pip install ultralytics opencv-python
"""

import sys
from datetime import datetime
import cv2
from pathlib import Path
from ultralytics import YOLO

from config import BEST_MODEL, SOURCE_VIDEO, CONFIDENCE


def make_output_path(source_video: Path) -> Path:
    """
    Build a unique output path of the form <stem>_<MMDDYYYY>_<NN>.mp4.

    The two-digit counter starts at 01 and increments until a filename that
    doesn't already exist on disk is found, so multiple runs on the same day
    are never overwritten.

    Args:
        source_video: Path to the input video (e.g. nuggets_thunder.mp4).

    Returns:
        A Path like nuggets_thunder_05172026_01.mp4 in the same directory.
    """
    date_str = datetime.now().strftime("%m%d%Y")
    stem     = source_video.stem          # "nuggets_thunder"
    counter  = 1
    while True:
        candidate = source_video.parent / f"{stem}_{date_str}_{counter:02d}.mp4"
        if not candidate.exists():
            return candidate
        counter += 1


def detect_logos(
    model_path: Path,
    source_video: Path,
    output_path: Path,
    confidence: float,
) -> None:
    """
    Detect logos in every frame of a video and write the annotated result to MP4.

    Args:
        model_path:   Path to trained YOLOv8 weights (.pt file).
        source_video: Path to the input video file.
        output_path:  Where to write the annotated output video (.mp4).
        confidence:   Minimum confidence score to display a detection (0–1).
    """
    # ── Validate inputs ────────────────────────────────────────────────────────
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}\nRun 4_train.py first.")
    if not source_video.exists():
        sys.exit(f"Source video not found: {source_video}")

    # ── Load the trained model ─────────────────────────────────────────────────
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # ── Open the source video ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {source_video}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Source: {source_video.name}  "
          f"({frame_width}×{frame_height} @ {fps:.1f} fps, {total_frames} frames)")
    print(f"Output: {output_path}")
    print(f"Confidence threshold: {confidence}")

    # ── Set up the MP4 writer ──────────────────────────────────────────────────
    # 'mp4v' is the codec tag for the MPEG-4 Part 2 codec.  Using VideoWriter
    # directly (instead of YOLO's built-in save=True) guarantees MP4 output on
    # every platform, whereas YOLO's default codec varies by OS.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (frame_width, frame_height)
    )

    # ── Process each frame ────────────────────────────────────────────────────
    print("\nProcessing frames...")
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Run inference on a single frame.
        # verbose=False suppresses the per-frame console output from YOLO.
        results = model.predict(frame, conf=confidence, verbose=False)

        # results[0].plot() returns a copy of the frame with bounding boxes,
        # class labels, and confidence scores drawn directly onto it.
        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

        frame_num += 1
        if frame_num % 100 == 0:
            pct = frame_num / total_frames * 100
            print(f"  {frame_num}/{total_frames} frames ({pct:.1f}%)")

    # ── Clean up ───────────────────────────────────────────────────────────────
    cap.release()
    writer.release()

    print(f"\nDone. Annotated video saved to:\n  {output_path}")


if __name__ == "__main__":
    output_path = make_output_path(SOURCE_VIDEO)
    detect_logos(BEST_MODEL, SOURCE_VIDEO, output_path, CONFIDENCE)
