"""
4_train.py — Step 4: Train a YOLOv8 logo detection model.

Loads the YOLOv8 Nano base weights, fine-tunes them on the annotated Bet365
logo dataset, and saves the best checkpoint to models/best.pt.

Training artifacts (loss curves, confusion matrix, sample predictions) are
written to runs/detect/<RUN_NAME>/ by the Ultralytics library.

Usage:
    python 4_train.py

Requirements:
    pip install ultralytics
"""

import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO

from config import (
    BASE_DIR, IMAGES_DIR, LABELS_DIR, MODELS_DIR,
    CLASS_NAMES, BASE_MODEL,
    EPOCHS, IMAGE_SIZE, BATCH_SIZE, PATIENCE, DEVICE, RUN_NAME,
)


def create_data_yaml() -> Path:
    """
    Write data.yaml with absolute paths for the current machine.

    YOLO requires absolute paths in data.yaml because it may be launched from
    any working directory.  We regenerate this file at runtime rather than
    committing a machine-specific copy to version control.

    Returns:
        Path to the written data.yaml file.
    """
    yaml_path = BASE_DIR / "data.yaml"
    config = {
        "path":  str(BASE_DIR),
        "train": str(IMAGES_DIR / "train"),
        "val":   str(IMAGES_DIR / "val"),
        "names": CLASS_NAMES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"data.yaml written to: {yaml_path}")
    return yaml_path


def train_model() -> None:
    """
    Fine-tune YOLOv8 on the annotated dataset and save the best model weights.

    After training completes, the best checkpoint is copied from the YOLO
    runs directory to models/best.pt so 5_detect.py always knows where to
    find it regardless of how many training runs have been executed.
    """
    yaml_path = create_data_yaml()

    # Start from the lightweight YOLOv8 Nano base model.
    # Swap BASE_MODEL for "yolov8s.pt", "yolov8m.pt", etc. in config.py for
    # higher accuracy at the cost of longer training and inference time.
    model = YOLO(BASE_MODEL)

    print(f"\nStarting training for up to {EPOCHS} epochs "
          f"(early stopping after {PATIENCE} epochs without improvement)...")

    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=DEVICE,
        name=RUN_NAME,
    )

    # ── Save best weights to a predictable location ────────────────────────────
    # Ultralytics writes the best checkpoint deep inside runs/detect/<name>/weights/.
    # Copying it to models/best.pt lets 5_detect.py find it without knowing the
    # exact run number or directory layout.
    best_from_run = BASE_DIR / "runs" / "detect" / RUN_NAME / "weights" / "best.pt"
    if best_from_run.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_from_run, MODELS_DIR / "best.pt")
        print(f"\nBest model saved to: {MODELS_DIR / 'best.pt'}")
    else:
        print(f"\nWarning: best.pt not found at {best_from_run}. "
              "Check the runs/ directory manually.")


if __name__ == "__main__":
    train_model()
