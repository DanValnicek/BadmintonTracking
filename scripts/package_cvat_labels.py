import json
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent

# Input paths
PREDICT_LABELS_DIR = BASE_DIR / "CourtDetection" / "runs" / "auto_annotations" / "batch_1"/"labels"
IMAGES_DIR = BASE_DIR / "CourtDetection" / "dataset" / "unfiltered_frames2"  # Location of your raw .jpg frames

# Output single JSON file path
OUTPUT_JSON_PATH = BASE_DIR / "CourtDetection" / "dataset" / "yolo_s_coco_annotations.json"

# Default resolution fallback if image files are stored elsewhere
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080

# Build COCO Keypoints Schema
coco_data = {
    "info": {"description": "Court Detection Auto-Annotations"},
    "licenses": [],
    "categories": [
        {
            "id": 1,
            "name": "court_skeleton",
            "supercategory": "court",
            "keypoints": [f"{i+1}" for i in range(28)],
            "skeleton": []
        }
    ],
    "images": [],
    "annotations": []
}

annotation_id = 1
label_files = sorted(list(PREDICT_LABELS_DIR.glob("*.txt")))

for image_id, txt_file in enumerate(label_files, start=1):
    image_filename = f"{txt_file.stem}.jpg"
    image_path = IMAGES_DIR / image_filename

    # Get exact frame dimensions to convert normalized coordinates
    if image_path.exists():
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    else:
        img_w, img_h = DEFAULT_WIDTH, DEFAULT_HEIGHT

    # Register Image reference
    coco_data["images"].append({
        "id": image_id,
        "file_name": image_filename,
        "width": img_w,
        "height": img_h
    })

    with open(txt_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(float, line.strip().split()))
        if len(parts) < 5:
            continue

        cls_id = int(parts[0])
        cx, cy, w, h = parts[1:5]

        # Convert YOLO bbox [cx, cy, w, h] to COCO absolute pixels [x_min, y_min, w, h]
        w_px = round(w * img_w, 2)
        h_px = round(h * img_h, 2)
        x_min = round((cx - w / 2) * img_w, 2)
        y_min = round((cy - h / 2) * img_h, 2)
        bbox = [x_min, y_min, w_px, h_px]

        # Convert keypoints to COCO format [x_px, y_px, v_code, ...]
        kpt_raw = parts[5:]
        coco_kpts = []
        num_visible = 0

        for i in range(0, len(kpt_raw), 3):
            kx = kpt_raw[i]
            ky = kpt_raw[i + 1]
            kv = kpt_raw[i + 2] if (i + 2) < len(kpt_raw) else 1.0

            kx_px = round(kx * img_w, 2)
            ky_px = round(ky * img_h, 2)

            # COCO visibility: 2 = visible, 0 = absent
            v_code = 2 if kv > 0.1 else 0
            if v_code > 0:
                num_visible += 1

            coco_kpts.extend([kx_px, ky_px, v_code])

        # Add annotation entry
        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": bbox,
            "area": round(w_px * h_px, 2),
            "keypoints": coco_kpts,
            "num_keypoints": num_visible,
            "iscrowd": 0,
            "segmentation": []
        })
        annotation_id += 1

# Save directly to single JSON file
with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(coco_data, f, indent=2)

print(f"Generated single annotation file: {OUTPUT_JSON_PATH}")