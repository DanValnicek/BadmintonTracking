import random
from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader, Sampler
from ultralytics import YOLO
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils.loss import v8PoseLoss
import torch
import torch.nn.functional as F
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.utils.loss import v8PoseLoss

BASE_DIR = Path(__file__).resolve().parent

# Load pretrained YOLOv8 Nano Pose weights
model = YOLO("../scripts/yolov8s-pose.pt")


def extract_video_id(img_path: str) -> str:
    # Splits '0p4Hvr_hycs_ts02m30s' -> '0p4Hvr_hycs'
    return Path(img_path).stem.split("_ts")[0]


class VideoStratifiedSampler(Sampler):
    """
    Balances video representation per epoch and yields indices in
    video-diverse chunks of `batch_size`.
    """

    def __init__(self, img_files, batch_size=8, max_frames_per_video=20):
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
                pools[vid] = shuffled[:self.target_per_video]
            else:
                repeated = shuffled * ((self.target_per_video // len(shuffled)) + 1)
                pools[vid] = repeated[:self.target_per_video]

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
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,
                drop_last=True,
                num_workers=self.args.workers,
                pin_memory=True,
                collate_fn=getattr(dataset, "collate_fn", None),
            )

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


def signed_perp_distance(p_start, p_end, p_query):
    """
    Signed perpendicular distance (in pixels) from p_query to the directed line p_start -> p_end.
    p_start, p_end: (N, 2)
    p_query: (N, M, 2)
    Returns: (N, M)
    """
    line_vec = p_end - p_start  # (N, 2)
    line_len = torch.linalg.norm(line_vec, dim=-1, keepdim=True).clamp(min=1e-6)  # (N, 1)
    rel = p_query - p_start.unsqueeze(1)  # (N, M, 2)

    # 2D cross product: dx * dy_q - dy * dx_q
    cross = line_vec[:, 0:1] * rel[..., 1] - line_vec[:, 1:2] * rel[..., 0]
    return cross / line_len


def court_geometry_loss(pred_xy, target_bboxes, geom_weight=0.05):
    """
    pred_xy: (N_fg, 28, 2) predicted keypoints in pixel coordinates
    target_bboxes: (N_fg, 4) ground-truth boxes (x1, y1, x2, y2) in pixel coordinates
    """
    if pred_xy.shape[0] == 0:
        return pred_xy.sum() * 0.0

    # Normalize pixel errors by court scale s = sqrt(w * h) so big/small courts weigh equally
    w = (target_bboxes[:, 2] - target_bboxes[:, 0]).clamp(min=1.0)
    h = (target_bboxes[:, 3] - target_bboxes[:, 1]).clamp(min=1.0)
    scale = torch.sqrt(w * h).unsqueeze(1)  # (N_fg, 1)

    collinear_losses = []
    ordering_losses = []

    # 1. Row collinearity: interior points of each row must lie on the line between row ends
    for row in COURT_ROWS:
        if len(row) >= 3:
            p_start = pred_xy[:, row[0]]
            p_end = pred_xy[:, row[-1]]
            p_mid = pred_xy[:, row[1:-1]]
            dist = signed_perp_distance(p_start, p_end, p_mid).abs() / scale
            collinear_losses.append(dist.mean())

    # 2. Column collinearity: interior points of each column must lie on the line between col ends
    for col in COURT_COLS:
        if len(col) >= 3:
            p_start = pred_xy[:, col[0]]
            p_end = pred_xy[:, col[-1]]
            p_mid = pred_xy[:, col[1:-1]]
            dist = signed_perp_distance(p_start, p_end, p_mid).abs() / scale
            collinear_losses.append(dist.mean())

    # 3. Row ordering: every point in subsequent rows must lie strictly in front of row r
    # (Assumes row[0] -> row[-1] goes left-to-right and rows go top-to-bottom in image space)
    for r_idx in range(len(COURT_ROWS) - 1):
        curr_row = COURT_ROWS[r_idx]
        p_start = pred_xy[:, curr_row[0]]
        p_end = pred_xy[:, curr_row[-1]]

        # All keypoints in rows below curr_row
        subsequent_indices = [k for r in COURT_ROWS[r_idx + 1:] for k in r]
        p_front = pred_xy[:, subsequent_indices]

        signed_dist = signed_perp_distance(p_start, p_end, p_front) / scale
        # Penalize if signed_dist <= 0 (point is behind or on top of the previous row)
        # Small margin (0.01 * scale) prevents rows from collapsing onto each other
        ordering_losses.append(F.relu(0.01 - signed_dist).mean())

    total_geom = torch.stack(collinear_losses + ordering_losses).mean()
    return geom_weight * total_geom


class GeometricPoseLoss(v8PoseLoss):
    def calculate_keypoints_loss(self, *args, **kwargs):
        kpts_loss, kpts_obj_loss = super().calculate_keypoints_loss(*args, **kwargs)

        masks = args[0]
        target_bboxes = args[-2]
        pred_kpts = args[-1]

        if masks.any():
            fg_pred_xy = pred_kpts[masks][..., :2]
            fg_bboxes = target_bboxes[masks]
            kpts_loss = kpts_loss + court_geometry_loss(fg_pred_xy, fg_bboxes, geom_weight=0.05)

        return kpts_loss, kpts_obj_loss

class GeometricPoseModel(PoseModel):
    def init_criterion(self):
        return GeometricPoseLoss(self)


class GeometricBalancedPoseTrainer(BalancedPoseTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(
            GeometricPoseModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                data_kpt_shape=self.data["kpt_shape"],
                verbose=verbose,
            )
        )
        if weights:
            model.load(weights)
        return model


results = model.train(
    trainer=GeometricBalancedPoseTrainer,
    data=str(BASE_DIR / "dataset" / "partially_annotated" / "data.yaml"),
    epochs=300,
    patience=50,
    imgsz=1280,
    batch=8,
    project=str(BASE_DIR / "runs"),
    name="v1_bootstrap",
    device=0,
    pose=24.0,
    fliplr=0.0,
    flipud=0.0,
    degrees=0.0,
    mosaic=0.0,
)
