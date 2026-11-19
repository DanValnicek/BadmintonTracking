import random
from collections import defaultdict
from pathlib import Path

import yaml
from torch.utils.data import DataLoader, Sampler
from ultralytics import YOLO
from ultralytics.models.yolo.pose import PoseTrainer

from court_geom import GeometricPoseModel

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "partially_annotated"
ORIG_YAML = DATASET_DIR / "data.yaml"


def extract_video_id(img_path: str | Path) -> str:
    return Path(img_path).stem.split("_ts")[0]


def create_video_grouped_split(
    dataset_dir: Path,
    orig_yaml_path: Path,
    num_val_videos: int = 1,
    seed: int = 42,
) -> Path:
    with open(orig_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    img_dir = dataset_dir / "images" / "train"
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in valid_exts)

    vid_to_imgs = defaultdict(list)
    for img_path in all_images:
        vid_to_imgs[extract_video_id(img_path)].append(img_path)

    all_vids = sorted(vid_to_imgs.keys())
    eligible_vids = [v for v in all_vids if len(vid_to_imgs[v]) >= 8] or all_vids

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
        f"Val: {len(val_imgs)} imgs across {len(val_vids)} unseen video(s) {sorted(val_vids)}"
    )
    return grouped_yaml_path


class VideoStratifiedSampler(Sampler):
    def __init__(self, img_files, batch_size=2, max_frames_per_video=20):
        self.batch_size = batch_size
        self.video_to_indices = defaultdict(list)

        for idx, path in enumerate(img_files):
            self.video_to_indices[extract_video_id(path)].append(idx)

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
            loader.reset = lambda: None
            return loader
        return super().get_dataloader(dataset_path, batch_size, rank, mode)


class GeometricBalancedPoseTrainer(BalancedPoseTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        model.__class__ = GeometricPoseModel
        return model


if __name__ == "__main__":
    grouped_yaml = create_video_grouped_split(
        dataset_dir=DATASET_DIR,
        orig_yaml_path=ORIG_YAML,
        num_val_videos=1,
        seed=42,
    )

    model = YOLO("yolov8l-pose.pt")

    results = model.train(
        trainer=GeometricBalancedPoseTrainer,
        data=str(grouped_yaml),
        epochs=300,
        patience=0,          # Do not stop before the 28-keypoint OKS cliff (~epoch 180)
        imgsz=1280,
        batch=3,
        optimizer="AdamW",
        lr0=0.0005,          # Prevents 44M-param Teacher from overfitting 16 videos
        lrf=0.01,
        weight_decay=0.001,
        warmup_epochs=5.0,
        project=str(BASE_DIR / "runs"),
        name="v4_teacher_large",
        device=0,
        pose=24.0,
        plots=True,
        close_mosaic=0,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        mosaic=0.0,
    )