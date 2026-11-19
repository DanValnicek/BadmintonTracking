from itertools import combinations
import torch
import torch.nn.functional as F
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils.loss import v8PoseLoss

# 1-indexed CVAT keypoint IDs (floor points 1..26; net-top 27 & 28 excluded)
COURT_ROWS_1IDX = [
    [1, 2, 3, 4, 5],        # Far baseline
    [6, 7, 8, 9, 10],       # Far doubles long service line
    [11, 12, 13],           # Far short service line
    [14, 15, 16],           # Near short service line
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

COURT_ROWS = [[k - 1 for k in row] for row in COURT_ROWS_1IDX]
COURT_COLS = [[k - 1 for k in col] for col in COURT_COLS_1IDX]


def _half_plane_order_loss(lines, pred_xy, gt_xy, gt_vis, scale, margin=0.01):
    """
    Enforces that every visible point in line[j] (j > i) lies on the exact same
    half-plane side of any visible segment (a, b) in line[i] as in ground truth.
    """
    losses = []
    for idx in range(len(lines) - 1):
        curr_line = lines[idx]
        subseq = [k for future_line in lines[idx + 1 :] for k in future_line]

        pairs = list(combinations(curr_line, 2))
        idx_a = [p[0] for p in pairs]
        idx_b = [p[1] for p in pairs]

        vis_pair = gt_vis[:, idx_a] & gt_vis[:, idx_b]
        vis_q = gt_vis[:, subseq]
        valid = vis_pair.unsqueeze(2) & vis_q.unsqueeze(1)
        if not valid.any():
            continue

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

        losses.append(F.relu(margin - gt_side * pred_signed)[valid].mean())
    return losses


def court_geometry_loss(pred_xy, gt_kpts, target_bboxes, geom_weight=0.10):
    """
    Visibility-aware collinearity + 2D grid ordering (both rows and columns).
    """
    if pred_xy.shape[0] == 0:
        return pred_xy.sum() * 0.0

    w = (target_bboxes[:, 2] - target_bboxes[:, 0]).clamp(min=1.0)
    h = (target_bboxes[:, 3] - target_bboxes[:, 1]).clamp(min=1.0)
    scale = torch.sqrt(w * h).unsqueeze(1)  # (N_fg, 1)

    gt_xy = gt_kpts[..., :2]
    gt_vis = gt_kpts[..., 2] > 0  # (N_fg, 28)

    collinear_losses = []

    # 1. Collinearity across all visible triplets on every row and column
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

    # 2. Row ordering (top-to-bottom) + Column ordering (left-to-right)
    row_order_losses = _half_plane_order_loss(COURT_ROWS, pred_xy, gt_xy, gt_vis, scale)
    col_order_losses = _half_plane_order_loss(COURT_COLS, pred_xy, gt_xy, gt_vis, scale)

    all_terms = collinear_losses + row_order_losses + col_order_losses
    if not all_terms:
        return pred_xy.sum() * 0.0

    return geom_weight * torch.stack(all_terms).mean()


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
                fg_pred_xy, fg_gt_kpts, fg_bboxes, geom_weight=0.10
            )

        return kpts_loss, kpts_obj_loss


class GeometricPoseModel(PoseModel):
    def init_criterion(self):
        return GeometricPoseLoss(self)