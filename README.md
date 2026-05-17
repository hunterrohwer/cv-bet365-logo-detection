# Bet365 Logo Detection in NBA Broadcast Footage

A custom object-detection pipeline that finds and tracks the **Bet365 sponsor logo**
in NBA game broadcast footage using a fine-tuned YOLOv8 model.

The project was developed against a Nuggets vs Thunder game recording and produces an
annotated output video (`Nuggets Thunder Result.mp4`) with a bounding box drawn around
the logo every time it appears on screen.

---

## Demo

The result of running the detection pipeline on `nuggets_thunder.mp4`:

| Original frame | Detected logo |
|---|---|
| Bet365 logo appears in the broadcast overlay | YOLOv8 draws a bounding box with confidence score |

To reproduce the result yourself:

```bash
# Install dependencies
pip install -r requirements.txt

# Run detection using the pre-trained weights (models/best.pt)
python 5_detect.py
```

This writes `Nuggets Thunder Result.mp4` to the project root.

---

## Project Structure

```
Logo Recognition Model/
│
├── config.py                 # ← Edit this to adapt for a new logo or video
│
├── 1_download_video.py       # Step 1: Download source footage from YouTube
├── 2_convert_annotations.py  # Step 2: Convert Label Studio JSON → YOLO format
├── 3_split_dataset.py        # Step 3: Split annotated images → train / val
├── 4_train.py                # Step 4: Fine-tune YOLOv8 on the dataset
├── 5_detect.py               # Step 5: Run detection, write annotated MP4
│
├── models/
│   └── best.pt               # Trained model weights (committed, ~6 MB)
│
├── images/                   # Training frames — not committed (see .gitignore)
│   ├── train/                #   259 PNG screenshots from the source video
│   └── val/                  #    65 PNG screenshots
│
├── labels/                   # YOLO label files — not committed (see .gitignore)
│   ├── train/                #   259 .txt files (one per training image)
│   └── val/                  #    65 .txt files
│
├── nuggets_thunder.mp4        # Source video — not committed (too large)
├── yolov8n.pt                 # YOLOv8 Nano base model
├── requirements.txt
└── .gitignore
```

---

## Full Pipeline — From Scratch

Follow these steps if you want to re-create the dataset and retrain the model
from the beginning. If you just want to run detection with the existing weights,
jump straight to [Quick Start](#demo).

### Step 1 — Download the source video

```bash
python 1_download_video.py
```

This downloads the game footage from YouTube and saves it as an MP4 file named
after the video title. Rename it to match `SOURCE_VIDEO` in `config.py` if needed.

### Step 2 — Capture frames for annotation

Open the source video in **VLC Media Player** and take snapshots of frames where
the Bet365 logo is visible (Video → Take Snapshot, or `Shift + S`).
Save all snapshots as PNG files into the `images/` directory.

Aim for 80–150 frames covering a variety of:
- Different logo sizes (near / far camera shots)
- Different on-screen positions
- Different lighting conditions and motion blur

### Step 3 — Annotate in Label Studio

Label Studio is a free, open-source annotation tool.

```bash
pip install label-studio
label-studio
```

1. Create a new project and upload the PNG images from `images/`.
2. Use the **Bounding Box** template and draw a box around the Bet365 logo in each image.
3. Export the project as **JSON** and save the file as `annotations.json` in the project root.

Then convert the export to YOLO format:

```bash
python 2_convert_annotations.py
```

This writes one `.txt` label file per image into `labels/`.

### Step 4 — Split the dataset

```bash
python 3_split_dataset.py
```

Randomly assigns 80 % of images to `images/train/` and 20 % to `images/val/`
(and their matching label files to `labels/train/` and `labels/val/`).

### Step 5 — Train the model

```bash
python 4_train.py
```

Fine-tunes YOLOv8 Nano on the annotated dataset for up to 50 epochs.
Training progress and metrics are saved to `runs/detect/logo_detection/`.
The best checkpoint is automatically copied to `models/best.pt`.

Training on CPU takes roughly 30–60 minutes depending on dataset size.
Set `DEVICE = "0"` in `config.py` to use a GPU.

### Step 6 — Detect logos in the video

```bash
python 5_detect.py
```

Processes the source video frame by frame and writes the annotated result to
`Nuggets Thunder Result.mp4`.

---

## Adapting for a New Project

Everything you need to change lives in **`config.py`**:

| Setting | What to change |
|---|---|
| `CLASS_NAMES` | Replace `"bet365_logo"` with your target logo name |
| `SOURCE_VIDEO` | Path to your input video |
| `OUTPUT_VIDEO` | Desired output filename |
| `YOUTUBE_URL` | URL of the source video on YouTube |
| `CONFIDENCE` | Lower for more detections, higher for fewer false positives |
| `BASE_MODEL` | Swap `yolov8n.pt` for `yolov8s.pt` / `m` / `l` / `x` for higher accuracy |
| `EPOCHS` | Increase if the model hasn't converged; decrease for a quick test |

No other files need to be edited.

---

## Requirements

```
ultralytics>=8.0   # YOLOv8 model and training framework
opencv-python>=4.8 # Video reading and writing
PyYAML>=6.0        # data.yaml generation
yt-dlp>=2024.0     # YouTube downloader (Step 1 only)
```

Install all at once:

```bash
pip install -r requirements.txt
```
