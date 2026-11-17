from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.tasks import PoseModel


# Stub required so torch.load can unpickle checkpoints trained with GeometricPoseModel
class GeometricPoseModel(PoseModel):
    pass


BASE_DIR = Path(__file__).resolve().parent

WEIGHTS_PATH = BASE_DIR / "runs" / "v1_bootstrap-27" / "weights" / "best.pt"
RAW_IMAGES_DIR = BASE_DIR / "dataset" / "unfiltered_frames2"
OUTPUT_DIR = BASE_DIR / "runs" / "auto_annotations" / "batch_1"
LABELS_DIR = OUTPUT_DIR / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(str(WEIGHTS_PATH))

kept, skipped = 0, 0
for r in model.predict(
    source=str(RAW_IMAGES_DIR),
    imgsz=1280,        # Must match training resolution (1280)
    conf=0.5,
    max_det=1,
    save=True,         # Saves visual preview images to check grid quality
    project=str(BASE_DIR / "runs" / "auto_annotations"),
    name="batch_1_previews",
    device=0,
    stream=True,
):
    stem = Path(r.path).stem
    if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
        skipped += 1
        continue

    kpt_xy = r.keypoints.xyn[0].cpu().numpy()     # (28, 2) normalized [0, 1]
    kpt_conf = r.keypoints.conf[0].cpu().numpy()  # (28,) confidence [0, 1]

    # Skip frames where average keypoint confidence is low so you don't fix scrambled grids
    if kpt_conf.mean() < 0.70:
        skipped += 1
        continue

    box = r.boxes.xywhn[0].cpu().numpy()
    tokens = ["0"] + [f"{v:.6f}" for v in box]
    for (x, y), c in zip(kpt_xy, kpt_conf):
        if c >= 0.50:
            tokens.extend([f"{x:.6f}", f"{y:.6f}", "2"])
        else:
            tokens.extend(["0.000000", "0.000000", "0"])

    (LABELS_DIR / f"{stem}.txt").write_text(" ".join(tokens) + "\n")
    kept += 1

print(f"Saved {kept} CVAT-ready label files to: {LABELS_DIR} (skipped {skipped} low-confidence frames)")