from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATASET_DIR = BASE_DIR / "CourtDetection" / "dataset" / "partially_annotated2"
LABELS_DIR = DATASET_DIR / "labels" / "train"

modified_files = 0
hidden_points_count = 0

for label_path in LABELS_DIR.glob("*.txt"):
    with open(label_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    file_modified = False

    for line in lines:
        parts = line.strip().split()
        if not parts or len(parts) < 5:
            continue

        class_id = parts[0]
        # Bounding box coordinates (clamp box to [0.0, 1.0])
        bbox = [max(0.0, min(1.0, float(v))) for v in parts[1:5]]
        bbox_str = [f"{v:.6f}" for v in bbox]

        # Keypoint triplets: [x1, y1, v1, x2, y2, v2, ...]
        kpt_tokens = parts[5:]
        new_kpts = []

        # Process in steps of 3 (x, y, visibility)
        for i in range(0, len(kpt_tokens), 3):
            if i + 2 >= len(kpt_tokens):
                break

            kx = float(kpt_tokens[i])
            ky = float(kpt_tokens[i + 1])
            kv = int(float(kpt_tokens[i + 2]))

            # Check if keypoint is out of frame boundary [0.0, 1.0]
            if kx < 0.0 or kx > 1.0 or ky < 0.0 or ky > 1.0:
                # Set coordinates to 0 and visibility flag to 0 (hidden/ignored)
                new_kpts.extend(["0.000000", "0.000000", "0"])
                hidden_points_count += 1
                file_modified = True
            else:
                new_kpts.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])

        new_line = f"{class_id} " + " ".join(bbox_str) + " " + " ".join(new_kpts) + "\n"
        new_lines.append(new_line)

    if file_modified:
        modified_files += 1
        with open(label_path, "w") as f:
            f.writelines(new_lines)

print(f"Finished processing label files!")
print(f"-> Modified {modified_files} files")
print(f"-> Marked {hidden_points_count} out-of-bounds keypoints as hidden (visibility = 0)")