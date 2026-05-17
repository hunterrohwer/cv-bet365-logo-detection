"""
3_split_dataset.py — Step 3: Split annotated images into training and validation sets.

Reads all PNG images from images/ and their matching YOLO label files from
labels/.  Randomly shuffles them, then moves 80 % to images/train/ &
labels/train/ and the remaining 20 % to the corresponding val/ folders.

Run this once after 2_convert_annotations.py has populated the labels/ folder.

Usage:
    python 3_split_dataset.py

Requirements:
    No extra packages — uses the Python standard library only.
"""

import random
import shutil
from pathlib import Path

from config import IMAGES_DIR, LABELS_DIR, TRAIN_SPLIT


def split_dataset() -> None:
    """
    Shuffle and split annotated images into train/ and val/ subdirectories.

    Images without a corresponding label file are skipped with a warning so
    a forgotten annotation never silently corrupts the training set.
    """
    # ── Gather all labelled images ─────────────────────────────────────────────
    # Only include images that have a matching label file — unlabelled images
    # would confuse the trainer by implying "no objects present here".
    all_images = sorted(IMAGES_DIR.glob("*.png"))
    labelled   = [img for img in all_images if (LABELS_DIR / f"{img.stem}.txt").exists()]

    skipped = len(all_images) - len(labelled)
    if skipped:
        print(f"Warning: {skipped} image(s) skipped — no matching label file found.")

    if not labelled:
        print("No labelled images found. Run 2_convert_annotations.py first.")
        return

    # ── Shuffle and split ──────────────────────────────────────────────────────
    random.shuffle(labelled)
    split_idx     = int(len(labelled) * TRAIN_SPLIT)
    train_images  = labelled[:split_idx]
    val_images    = labelled[split_idx:]

    # ── Create output directories ──────────────────────────────────────────────
    for split in ("train", "val"):
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

    # ── Move files ─────────────────────────────────────────────────────────────
    def move_pair(img: Path, split: str) -> None:
        """Move one image and its label into the appropriate split folder."""
        shutil.move(str(img),               str(IMAGES_DIR / split / img.name))
        shutil.move(str(LABELS_DIR / f"{img.stem}.txt"),
                    str(LABELS_DIR / split  / f"{img.stem}.txt"))

    for img in train_images:
        move_pair(img, "train")
    for img in val_images:
        move_pair(img, "val")

    print(f"Dataset split complete.")
    print(f"  Training:   {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")


if __name__ == "__main__":
    split_dataset()
