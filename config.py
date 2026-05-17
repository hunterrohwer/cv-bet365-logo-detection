"""
config.py — Central configuration for the Bet365 Logo Detection pipeline.

This is the only file you need to edit when adapting this project for a
different logo or video. Every other script imports its settings from here.
"""

from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────────────────
# BASE_DIR resolves to wherever this file lives, so no hardcoded paths anywhere.
BASE_DIR   = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images"
LABELS_DIR = BASE_DIR / "labels"
MODELS_DIR = BASE_DIR / "models"

# ─── Dataset Configuration ────────────────────────────────────────────────────
# Map each integer class ID to a human-readable label name.
# Add more entries (e.g. {0: "logo_a", 1: "logo_b"}) for multi-class projects.
CLASS_NAMES = {0: "bet365_logo"}

# Fraction of images used for training; the remainder go to validation.
TRAIN_SPLIT = 0.8

# ─── Training Configuration ───────────────────────────────────────────────────
# Which YOLOv8 variant to start from.
# Options (smallest → largest, faster → more accurate): n, s, m, l, x
BASE_MODEL = "yolov8n.pt"

EPOCHS     = 50    # Maximum training epochs (early stopping may end it sooner)
IMAGE_SIZE = 640   # Input resolution fed to the network (pixels)
BATCH_SIZE = 8     # Images processed per gradient step
PATIENCE   = 20    # Stop early if validation loss doesn't improve for this many epochs
DEVICE     = "cpu" # Use "0" (or "cuda") to train on GPU instead
RUN_NAME   = "logo_detection"

# ─── Inference Configuration ──────────────────────────────────────────────────
# Video to run detection on.
SOURCE_VIDEO = BASE_DIR / "nuggets_thunder.mp4"

# Trained model weights produced by 4_train.py.
BEST_MODEL = MODELS_DIR / "best.pt"

# Only draw bounding boxes when the model is at least this confident (0–1).
CONFIDENCE = 0.3

# ─── Download Configuration ───────────────────────────────────────────────────
# YouTube URL of the source game footage.
YOUTUBE_URL = "https://www.youtube.com/watch?v=Yl08L6Y0y3Y"
