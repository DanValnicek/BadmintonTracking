from pathlib import Path
import shutil

# Project roots using relative paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_IMG_DIR = BASE_DIR / "dataset" / "unfiltered_frames2"
CVAT_LABEL_DIR = BASE_DIR / "dataset" / "partially_annotated" / "labels" / "train"

DEST_IMG_DIR = BASE_DIR / "dataset" / "partially_annotated" / "images" / "train"
DEST_LBL_DIR = BASE_DIR / "dataset" / "partially_annotated" / "labels" / "train"

DEST_IMG_DIR.mkdir(parents=True, exist_ok=True)
DEST_LBL_DIR.mkdir(parents=True, exist_ok=True)

copied = 0
for txt_file in CVAT_LABEL_DIR.glob("*.txt"):
    # Skip empty files if any exist
    if txt_file.stat().st_size == 0:
        continue

    jpg_name = f"{txt_file.stem}.jpg"
    src_jpg = RAW_IMG_DIR / jpg_name
    dst_jpg = DEST_IMG_DIR / jpg_name
    dst_txt = DEST_LBL_DIR / txt_file.name

    # Copy image if source exists and paths are not the same
    if src_jpg.exists() and src_jpg.resolve() != dst_jpg.resolve():
        shutil.copy(src_jpg, dst_jpg)

    # Copy label file only if source and destination paths are different
    if txt_file.resolve() != dst_txt.resolve():
        shutil.copy(txt_file, dst_txt)

    copied += 1

print(f"Prepared {copied} image-label pairs for training.")