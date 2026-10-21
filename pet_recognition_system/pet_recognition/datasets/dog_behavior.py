"""Dog-behaviour dataset backed by generated RTMPose NPZ files."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pet_recognition.preprocessing.skeleton import (
    apply_temporal_indices,
    augment_skeleton,
    extract_box_motion,
    extract_global_motion,
    interpolate_missing,
    normalize_dog_skeleton,
    temporal_indices,
)

DOG_BEHAVIOR_CLASSES = (
    "walking_run", "standing", "sitting", "lying_sleeping", "eating_drinking"
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(DOG_BEHAVIOR_CLASSES)}


def read_behavior_manifest(path: str | Path, split: str | None = None) -> list[dict[str, str]]:
    """Read valid rows and optionally select one frozen split."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"clip_id", "dog_id", "behavior_label", "split", "valid"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    valid = [row for row in rows if row["valid"] == "1"]
    unknown = sorted({row["behavior_label"] for row in valid} - set(DOG_BEHAVIOR_CLASSES))
    if unknown:
        raise ValueError(f"{path}: unknown behaviour labels {unknown}")
    if split is not None:
        valid = [row for row in valid if row["split"] == split]
    return valid


class DogBehaviorSkeletonDataset(Dataset):
    """Load the dog stream and return ST-GCN input ``(C,T,V,M)``."""

    def __init__(self, manifest, skeleton_dir, split, *, target_frames=32, training=False,
                 augment=True, include_global_motion=True, seed=28):
        self.rows = read_behavior_manifest(manifest, split)
        if not self.rows:
            raise ValueError(f"no valid rows found for split={split!r}; freeze the split first")
        self.skeleton_dir = Path(skeleton_dir)
        self.target_frames = target_frames
        self.training = training
        self.augment = augment and training
        self.include_global_motion = include_global_motion
        self.seed = seed
        self.epoch = 0
        missing = [row["clip_id"] for row in self.rows if not self._path(row["clip_id"]).exists()]
        if missing:
            raise FileNotFoundError(f"missing {len(missing)} skeleton files; first: {missing[:5]}")

    def _path(self, clip_id: str) -> Path:
        return self.skeleton_dir / f"dog_{clip_id}.npz"

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        with np.load(self._path(row["clip_id"]), allow_pickle=False) as archive:
            skeletons = archive["skeletons"]
            if skeletons.ndim != 4 or skeletons.shape[0] != 2 or skeletons.shape[2:] != (17, 3):
                raise ValueError(f"{row['clip_id']}: expected skeletons (2,T,17,3), got {skeletons.shape}")
            dog = skeletons[1].astype(np.float32, copy=True)
            image_size = archive["image_size"]
            boxes = archive["boxes"] if "boxes" in archive else None
        dog = interpolate_missing(dog)
        if boxes is not None and boxes.ndim == 3 and boxes.shape[:2] == skeletons.shape[:2]:
            global_motion = extract_box_motion(boxes[1], image_size)
        else:
            global_motion = extract_global_motion(dog, image_size)
        dog = normalize_dog_skeleton(dog, image_size)
        rng = np.random.default_rng(self.seed + self.epoch * 100_003 + index)
        indices, frame_mask = temporal_indices(
            dog.shape[0], self.target_frames, training=self.training, rng=rng
        )
        dog = apply_temporal_indices(dog, indices, frame_mask, self.target_frames)
        global_motion = apply_temporal_indices(
            global_motion, indices, frame_mask, self.target_frames
        )
        if self.augment:
            dog, global_motion = augment_skeleton(dog, rng, global_motion)
        if not self.include_global_motion:
            global_motion.fill(0)
        data = np.ascontiguousarray(dog.transpose(2, 0, 1)[..., None])
        return {
            "keypoints": torch.from_numpy(data),
            "frame_mask": torch.from_numpy(frame_mask),
            "global_motion": torch.from_numpy(np.ascontiguousarray(global_motion)),
            "label": torch.tensor(CLASS_TO_INDEX[row["behavior_label"]], dtype=torch.long),
            "clip_id": row["clip_id"],
            "dog_id": row["dog_id"],
        }


def class_counts(rows: list[dict[str, str]]) -> np.ndarray:
    counts = np.zeros(len(DOG_BEHAVIOR_CLASSES), dtype=np.int64)
    for row in rows:
        counts[CLASS_TO_INDEX[row["behavior_label"]]] += 1
    return counts
