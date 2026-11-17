import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Sampler
from ultralytics import YOLO
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils.loss import v8PoseLoss

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "partially_annotated"
ORIG_YAML = DATASET_DIR / "data.yaml"


def extract_video_id(img_path: str | Path) -> str:
    # Splits '0p4Hvr_hycs_ts02m30s' -> '0p4Hvr_hycs'
    return Path(img_path).stem.split("_ts")[0]


def create_video_grouped_split(
    dataset_dir: Path,
    orig_yaml_path: Path,
    num_val_videos: int = 2,
    seed: int = 42,
) -> Path:
    """
    Holds out `num_val_videos` entire YouTube videos for validation and writes
    train.txt, val.txt, and data_grouped.yaml without moving any files on disk.
    """
    with open(orig_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    img_dir = dataset_dir / "images" / "train"
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in valid_exts)

    vid_to_imgs = defaultdict(list)
    for img_path in all_images:
        vid_to_imgs[extract_video_id(img_path)].append(img_path)

    all_vids = sorted(vid_to_imgs.keys())
    # Prefer holding out videos with at least 5 frames so validation mAP is meaningful
    eligible_vids = [v for v in all_vids if len(vid_to_imgs[v]) >= 5]
    if len(eligible_vids) < num_val_videos:
        eligible_vids = all_vids

    rng = random.Random(seed)
    val_vids = set(rng.sample(eligible_vids, k=min(num_val_videos, len(eligible_vids))))

    train_imgs, val_imgs = [], []
    for vid, imgs in vid_to_imgs.items():
        if vid in val_vids:
            val_imgs.extend(imgs)
        else:
            train_imgs.extend(imgs)

    train_txt = dataset_dir / "train.txt"
    val_txt = dataset_dir / "val.txt"
    train_txt.write_text("\n".join(str(p.resolve()) for p in train_imgs) + "\n")
    val_txt.write_text("\n".join(str(p.resolve()) for p in val_imgs) + "\n")

    grouped_cfg = dict(cfg)
    grouped_cfg.pop("path", None)
    grouped_cfg["train"] = str(train_txt.resolve())
    grouped_cfg["val"] = str(val_txt.resolve())

    grouped_yaml_path = dataset_dir / "data_grouped.yaml"
    with open(grouped_yaml_path, "w") as f:
        yaml.safe_dump(grouped_cfg, f, sort_keys=False)

    print(
        f"[Split] Train: {len(train_imgs)} imgs across {len(all_vids) - len(val_vids)} videos | "
        f"Val: {len(val_imgs)} imgs across {len(val_vids)} unseen videos {sorted(val_vids)}"
    )
    return grouped_yaml_path


class VideoStratifiedSampler(Sampler):
    """
    Balances video representation per epoch and yields indices in
    video-diverse chunks of `batch_size`.
    """

    def __init__(self, img_files, batch_size=2, max_frames_per_video=20):
        self.batch_size = batch_size
        self.video_to_indices = defaultdict(list)

        for idx, path in enumerate(img_files):
            vid = extract_video_id(path)
            self.video_to_indices[vid].append(idx)

        max_existing = max(len(idxs) for idxs in self.video_to_indices.values())
        self.target_per_video = min(max_existing, max_frames_per_video)
        self.num_videos = len(self.video_to_indices)
        self.num_batches = (self.num_videos * self.target_per_video) // self.batch_size

    def __iter__(self):
        pools = {}
        for vid, idxs in self.video_to_indices.items():
            shuffled = idxs.copy()
            random.shuffle(shuffled)

            if len(shuffled) >= self.target_per_video:
                pools[vid] = shuffled[: self.target_per_video]
            else:
                repeated = shuffled * ((self.target_per_video // len(shuffled)) + 1)
                pools[vid] = repeated[: self.target_per_video]

        for _ in range(self.num_batches):
            active_vids = [v for v, p in pools.items() if len(p) > 0]
            active_vids.sort(key=lambda v: len(pools[v]), reverse=True)

            batch = []
            while len(batch) < self.batch_size and active_vids:
                for vid in active_vids:
                    if pools[vid] and len(batch) < self.batch_size:
                        batch.append(pools[vid].pop())
                active_vids = [v for v in active_vids if len(pools[v]) > 0]

            if len(batch) == self.batch_size:
                yield from batch

    def __len__(self):
        return self.num_batches * self.batch_size


class BalancedPoseTrainer(PoseTrainer):
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        dataset = self.build_dataset(dataset_path, mode, batch_size)

        if mode == "train":
            sampler = VideoStratifiedSampler(
                dataset.im_files,
                batch_size=batch_size,
                max_frames_per_video=20,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,
                drop_last=True,
                num_workers=self.args.workers,
                pin_memory=True,
                collate_fn=getattr(dataset, "collate_fn", None),
            )
            # Prevent AttributeError if Ultralytics calls train_loader.reset()
            loader.reset = lambda: None
            return loader

        return super().get_dataloader(dataset_path, batch_size, rank, mode)


# 1-indexed CVAT keypoint IDs (floor points 1..26 only; 27 & 28 net-top points excluded)
COURT_ROWS_1IDX = [
    [1, 2, 3, 4, 5],        # Far baseline
    [6, 7, 8, 9, 10],       # Far doubles long service line
    [11, 12, 13],           # Far short service line (doubles-L, T-junction, doubles-R)
    [14, 15, 16],           # Near short service line (doubles-L, T-junction, doubles-R)
    [17, 18, 19, 20, 21],   # Near doubles long service line
    [22, 23, 24, 25, 26],   # Near baseline
]

COURT_COLS_1IDX = [
    [1, 6, 11, 14, 17, 22],   # Left doubles sideline
    [2, 7, 18, 23],           # Left singles sideline
    [3, 8, 12, 15, 19, 24],   # Center line
    [4, 9, 20, 25],           # Right singles sideline
    [5, 10, 13, 16, 21, 26],  # Right doubles sideline
]

# Convert to 0-indexed PyTorch tensor indices (0..25)
COURT_ROWS = [[k - 1 for k in row] for row in COURT_ROWS_1IDX]
COURT_COLS = [[k - 1 for k in col] for col in COURT_COLS_1IDX]


def court_geometry_loss(pred_xy, gt_kpts, target_bboxes, geom_weight=0.05):
    """
    Enforces collinearity and row ordering strictly across visible (v > 0) keypoints.
    Invisible points (v == 0) are completely ignored as both anchors and query points.
    """
    if pred_xy.shape[0] == 0:
        return pred_xy.sum() * 0.0

    w = (target_bboxes[:, 2] - target_bboxes[:, 0]).clamp(min=1.0)
    h = (target_bboxes[:, 3] - target_bboxes[:, 1]).clamp(min=1.0)
    scale = torch.sqrt(w * h).unsqueeze(1)  # (N_fg, 1)

    gt_xy = gt_kpts[..., :2]
    gt_vis = gt_kpts[..., 2] > 0  # (N_fg, 28)

    collinear_losses = []
    ordering_losses = []

    # 1. Collinearity across all visible triplets (a, b, c) on every row & column
    for line in COURT_ROWS + COURT_COLS:
        if len(line) < 3:
            continue
        triplets = list(combinations(line, 3))
        idx_a = [t[0] for t in triplets]
        idx_b = [t[1] for t in triplets]
        idx_c = [t[2] for t in triplets]

        vis_triplet = gt_vis[:, idx_a] & gt_vis[:, idx_b] & gt_vis[:, idx_c]
        if not vis_triplet.any():
            continue

        p_a = pred_xy[:, idx_a]
        p_b = pred_xy[:, idx_b]
        p_c = pred_xy[:, idx_c]

        line_vec = p_c - p_a
        line_len = torch.linalg.norm(line_vec, dim=-1).clamp(min=1e-6)
        rel = p_b - p_a

        cross = (line_vec[..., 0] * rel[..., 1] - line_vec[..., 1] * rel[..., 0]).abs()
        dist = (cross / line_len) / scale
        collinear_losses.append(dist[vis_triplet].mean())

    # 2. Row ordering across visible anchor pairs (a, b) and visible subsequent points q
    for r_idx in range(len(COURT_ROWS) - 1):
        curr_row = COURT_ROWS[r_idx]
        subseq = [k for r in COURT_ROWS[r_idx + 1 :] for k in r]

        pairs = list(combinations(curr_row, 2))
        idx_a = [p[0] for p in pairs]
        idx_b = [p[1] for p in pairs]

        vis_pair = gt_vis[:, idx_a] & gt_vis[:, idx_b]
        vis_q = gt_vis[:, subseq]
        valid = vis_pair.unsqueeze(2) & vis_q.unsqueeze(1)
        if not valid.any():
            continue

        # Ground-truth half-plane side (+1 or -1) so angled/elevated cameras never fight the loss
        gt_a = gt_xy[:, idx_a].unsqueeze(2)
        gt_b = gt_xy[:, idx_b].unsqueeze(2)
        gt_q = gt_xy[:, subseq].unsqueeze(1)
        gt_vec = gt_b - gt_a
        gt_rel = gt_q - gt_a
        gt_side = torch.sign(gt_vec[..., 0] * gt_rel[..., 1] - gt_vec[..., 1] * gt_rel[..., 0])

        valid = valid & (gt_side != 0)
        if not valid.any():
            continue

        pred_a = pred_xy[:, idx_a].unsqueeze(2)
        pred_b = pred_xy[:, idx_b].unsqueeze(2)
        pred_q = pred_xy[:, subseq].unsqueeze(1)
        pred_vec = pred_b - pred_a
        pred_len = torch.linalg.norm(pred_vec, dim=-1).clamp(min=1e-6)
        pred_rel = pred_q - pred_a
        pred_cross = pred_vec[..., 0] * pred_rel[..., 1] - pred_vec[..., 1] * pred_rel[..., 0]
        pred_signed = (pred_cross / pred_len) / scale.unsqueeze(-1)

        ordering_losses.append(F.relu(0.01 - gt_side * pred_signed)[valid].mean())

    if not collinear_losses and not ordering_losses:
        return pred_xy.sum() * 0.0

    total_geom = torch.stack(collinear_losses + ordering_losses).mean()
    return geom_weight * total_geom


class GeometricPoseLoss(v8PoseLoss):
    def calculate_keypoints_loss(self, *args, **kwargs):
        kpts_loss, kpts_obj_loss = super().calculate_keypoints_loss(*args, **kwargs)

        masks = args[0]          # (B, num_anchors)
        target_gt_idx = args[1]  # (B, num_anchors)
        keypoints = args[2]      # (B, max_obj, 28, 3)
        target_bboxes = args[-2] # (B, num_anchors, 4)
        pred_kpts = args[-1]     # (B, num_anchors, 28, 3)

        if masks.any():
            fg_pred_xy = pred_kpts[masks][..., :2]
            fg_bboxes = target_bboxes[masks]

            if keypoints.ndim == 4:
                bsz, num_anchors = masks.shape
                batch_idx = torch.arange(bsz, device=masks.device).unsqueeze(1).expand(bsz, num_anchors)
                fg_gt_kpts = keypoints[batch_idx[masks], target_gt_idx[masks]]
            else:
                fg_gt_kpts = keypoints[target_gt_idx[masks]]

            kpts_loss = kpts_loss + court_geometry_loss(
                fg_pred_xy, fg_gt_kpts, fg_bboxes, geom_weight=0.05
            )

        return kpts_loss, kpts_obj_loss
class GeometricPoseModel(PoseModel):
    def init_criterion(self):
        return GeometricPoseLoss(self)


class GeometricBalancedPoseTrainer(BalancedPoseTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        model.__class__ = GeometricPoseModel
        return model


if __name__ == "__main__":
    grouped_yaml = create_video_grouped_split(
        dataset_dir=DATASET_DIR,
        orig_yaml_path=ORIG_YAML,
        num_val_videos=2,
        seed=42,
    )

    # Auto-downloads yolov8l-pose.pt if not already present locally
    model = YOLO("yolov8l-pose.pt")

    results = model.train(
        trainer=GeometricBalancedPoseTrainer,
        data=str(grouped_yaml),
        epochs=300,
        patience=60,
        imgsz=1280,
        batch=3,             # Safe for yolov8l-pose at 1280px on 8GB RTX 4060
        project=str(BASE_DIR / "runs"),
        name="v3_large_geom_valsplit",
        device=0,
        pose=24.0,
        plots=True,
        close_mosaic=0,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        mosaic=0.0,
    )