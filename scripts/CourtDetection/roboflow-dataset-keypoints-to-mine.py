import shutil
from pathlib import Path
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RF_ROOT = BASE_DIR / "CourtDetection" / "dataset" / "roboflow-dataset"
OUT_IMG_DIR = BASE_DIR / "CourtDetection" / "dataset" / "roboflow_28kpts" / "images"
OUT_LBL_DIR = BASE_DIR / "CourtDetection" / "dataset" / "roboflow_28kpts" / "labels"

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

# Verified mapping: our 0-indexed floor keypoint (0..25) -> Roboflow 0-indexed keypoint (0..29)
OUR_TO_RF = [
    0, 8, 9, 10, 3,  # Row 1: Far baseline (pt_1..pt_5)
    4, 11, 12, 13, 14,  # Row 2: Far doubles long service line (pt_6..pt_10)
    5, 18, 15,  # Row 3: Far short service line (pt_11..pt_13)
    6, 28, 16,  # Row 4: Near short service line (pt_14..pt_16)
    7, 26, 25, 21, 17,  # Row 5: Near doubles long service line (pt_17..pt_21)
    1, 23, 24, 22, 2,  # Row 6: Near baseline (pt_22..pt_26)
]

converted = 0
saved_preview = False

for split in ("train", "valid", "test"):
    img_dir = RF_ROOT / split / "images"
    lbl_dir = RF_ROOT / split / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        continue

    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        # Find matching image (.jpg / .png)
        matches = list(img_dir.glob(f"{lbl_path.stem}.*"))
        if not matches:
            continue
        img_path = matches[0]

        lines = [ln.strip() for ln in lbl_path.read_text().splitlines() if ln.strip()]
        if not lines:
            continue

        out_lines = []
        preview_pts = []

        for ln in lines:
            parts = list(map(float, ln.split()))
            if len(parts) < 5 + 30 * 3:
                continue

            cls_id = int(parts[0])
            bbox = parts[1:5]
            rf_raw = parts[5: 5 + 30 * 3]
            rf_kpts = [rf_raw[i: i + 3] for i in range(0, len(rf_raw), 3)]

            tokens = [str(cls_id)] + [f"{v:.6f}" for v in bbox]

            # Map 26 floor keypoints
            for our_idx, rf_idx in enumerate(OUR_TO_RF):
                kx, ky, kv = rf_kpts[rf_idx]
                vis = int(round(kv))
                if vis > 0:
                    tokens.extend([f"{kx:.6f}", f"{ky:.6f}", "2"])
                    preview_pts.append((our_idx + 1, kx, ky))
                else:
                    tokens.extend(["0.000000", "0.000000", "0"])

            # Pad net-top keypoints pt_27 & pt_28 as absent (v=0)
            tokens.extend(["0.000000", "0.000000", "0"] * 2)
            out_lines.append(" ".join(tokens))

        if not out_lines:
            continue

        new_stem = f"rf_{split}_{img_path.stem}"
        dst_img = OUT_IMG_DIR / f"{new_stem}{img_path.suffix.lower()}"
        dst_lbl = OUT_LBL_DIR / f"{new_stem}.txt"

        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text("\n".join(out_lines) + "\n")
        converted += 1

        if not saved_preview and preview_pts:
            with Image.open(dst_img).convert("RGB") as im:
                w, h = im.size
                draw = ImageDraw.Draw(im)
                for pt_num, kx, ky in preview_pts:
                    px, py = kx * w, ky * h
                    draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(0, 255, 0))
                    draw.text((px + 5, py - 5), str(pt_num), fill=(255, 255, 0))
                im.save(OUT_IMG_DIR.parent / "preview_check.jpg")
            saved_preview = True

print(f"Converted {converted} Roboflow images into {OUT_IMG_DIR.parent}")
