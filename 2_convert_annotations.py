"""
2_convert_annotations.py — Step 2: Convert Label Studio JSON exports to YOLO format.

Label Studio exports bounding-box coordinates as percentages of the image
dimensions (x, y, width, height — each 0–100). YOLO expects normalised values
(0–1) expressed as (center_x, center_y, width, height).

This script reads the JSON file you exported from Label Studio and writes one
.txt label file per image into the labels/ directory.  Each line in a label
file follows the YOLO format:

    <class_id> <center_x> <center_y> <width> <height>

All four coordinate values are normalised to [0, 1].

Usage:
    1. In Label Studio, export your project as "JSON".
    2. Save the exported file as "annotations.json" in this project directory.
    3. Run:  python 2_convert_annotations.py

Requirements:
    No extra packages — uses the Python standard library only.
"""

import json
from pathlib import Path

from config import BASE_DIR, LABELS_DIR


def convert_annotations(json_path: Path, output_dir: Path) -> None:
    """
    Convert a Label Studio JSON export to per-image YOLO .txt label files.

    Args:
        json_path:  Path to the Label Studio JSON export file.
        output_dir: Directory where the YOLO .txt files will be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load the Label Studio export ──────────────────────────────────────────
    print(f"Reading annotations from: {json_path}")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {json_path} not found.")
        print("Export your Label Studio project as JSON and save it as 'annotations.json'.")
        return
    except json.JSONDecodeError:
        print("ERROR: annotations.json is not valid JSON. Re-export from Label Studio.")
        return

    print(f"Processing {len(data)} annotated images...")
    converted_count = 0

    # ── Process each annotated image ──────────────────────────────────────────
    for item in data:
        try:
            # Resolve the original image filename from the Label Studio task data.
            image_path = item.get("data", {}).get("image", "")
            if not image_path:
                print(f"  Warning: no image path found in item, skipping.")
                continue
            image_stem = Path(image_path).stem  # filename without extension

            # Each image may have multiple annotation results (one per bounding box).
            annotations = item.get("annotations", [])
            if not annotations:
                continue

            yolo_lines = []
            for result in annotations[0].get("result", []):
                value = result.get("value", {})
                # Skip results that aren't bounding boxes.
                if "width" not in value or "height" not in value:
                    continue

                # Label Studio stores coordinates as percentages (0–100).
                # Convert to normalised fractions (0–1) required by YOLO.
                x_pct = value["x"] / 100.0
                y_pct = value["y"] / 100.0
                w     = value["width"]  / 100.0
                h     = value["height"] / 100.0

                # YOLO needs the box center, but Label Studio gives the top-left corner.
                cx = x_pct + w / 2
                cy = y_pct + h / 2

                # Class 0 is the only class in this project ("bet365_logo").
                # For multi-class projects, map the label name to its class ID here.
                class_id = 0
                yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # Write a label file only if at least one box was found.
            if yolo_lines:
                out_file = output_dir / f"{image_stem}.txt"
                out_file.write_text("\n".join(yolo_lines))
                converted_count += 1

        except Exception as e:
            print(f"  Error processing item: {e}")

    print(f"\nConversion complete. {converted_count} label files written to: {output_dir}")


if __name__ == "__main__":
    json_path = BASE_DIR / "annotations.json"
    convert_annotations(json_path, LABELS_DIR)
