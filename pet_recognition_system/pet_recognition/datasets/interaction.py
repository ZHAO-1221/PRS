"""Human–dog interaction dataset backed by synchronized skeleton NPZ files."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from pet_recognition.preprocessing.skeleton import (
    apply_temporal_indices, extract_relation_features, interpolate_missing,
    normalize_dog_skeleton, normalize_human_skeleton, temporal_indices,
)

INTERACTION_CLASSES = ("no_interaction", "hand_feeding", "petting", "playing_together")
CLASS_TO_INDEX = {name: index for index, name in enumerate(INTERACTION_CLASSES)}


def read_interaction_manifest(path, split=None):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"clip_id", "dog_id", "person_id", "interaction", "split", "valid"}
    missing = required - set(rows[0] if rows else ())
    if missing: raise ValueError(f"{path}: missing columns {sorted(missing)}")
    rows = [row for row in rows if row["valid"] == "1"]
    unknown = sorted({row["interaction"] for row in rows} - set(INTERACTION_CLASSES))
    if unknown: raise ValueError(f"unknown interaction labels {unknown}")
    return [row for row in rows if split is None or row["split"] == split]


class InteractionSkeletonDataset(Dataset):
    def __init__(self, manifest, skeleton_dir, split, *, target_frames=32,
                 training=False, strong_augmentation=False, seed=28):
        self.rows = read_interaction_manifest(manifest, split)
        if not self.rows: raise ValueError(f"no valid rows for split={split!r}")
        self.skeleton_dir = Path(skeleton_dir)
        self.target_frames = target_frames
        self.training = training
        self.strong_augmentation = strong_augmentation
        self.seed = seed
        self.epoch = 0
        missing = [r["clip_id"] for r in self.rows if not self._path(r["clip_id"]).exists()]
        if missing: raise FileNotFoundError(f"missing {len(missing)} skeleton files; first {missing[:5]}")

    def _path(self, clip_id): return self.skeleton_dir / f"interaction_{clip_id}.npz"
    def __len__(self): return len(self.rows)
    def set_epoch(self, epoch): self.epoch = epoch

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(self._path(row["clip_id"]), allow_pickle=False) as archive:
            skeletons = archive["skeletons"].astype(np.float32, copy=True)
            boxes = archive["boxes"].astype(np.float32, copy=True)
            image_size = archive["image_size"]
        if skeletons.ndim != 4 or skeletons.shape[0] != 2 or skeletons.shape[2:] != (17, 3):
            raise ValueError(f"{row['clip_id']}: invalid skeleton shape {skeletons.shape}")
        rng = np.random.default_rng(self.seed + self.epoch * 100_003 + index)
        if self.training and self.strong_augmentation:
            # Geometry-preserving mirror with anatomical left/right swaps.
            if rng.random() < 0.5:
                skeletons[..., 0] = float(image_size[0]) - skeletons[..., 0]
                for subject, pairs in enumerate((
                    ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
                     (11, 12), (13, 14), (15, 16)),
                    ((0, 1), (5, 8), (6, 9), (7, 10),
                     (11, 14), (12, 15), (13, 16)),
                )):
                    for left, right in pairs:
                        skeletons[subject, :, (left, right)] = skeletons[subject, :, (right, left)]
                old_left = boxes[..., 0].copy()
                boxes[..., 0] = float(image_size[0]) - boxes[..., 2]
                boxes[..., 2] = float(image_size[0]) - old_left
            # Small pose-estimation noise plus one short missing-joint event.
            visible = skeletons[..., 2] > 0
            noise = rng.normal(0, max(float(image_size.max()) * 0.002, 0.5),
                               size=skeletons[..., :2].shape).astype(np.float32)
            skeletons[..., :2][visible] += noise[visible]
            if rng.random() < 0.7:
                subject = int(rng.integers(0, 2)); joint = int(rng.integers(0, 17))
                length = max(1, int(round(len(skeletons[subject]) * rng.uniform(0.05, 0.18))))
                start = int(rng.integers(0, max(len(skeletons[subject]) - length + 1, 1)))
                skeletons[subject, start:start + length, joint] = 0
        human, dog = interpolate_missing(skeletons[0]), interpolate_missing(skeletons[1])
        relation_human, relation_dog = skeletons[0].copy(), skeletons[1].copy()
        relation_human[relation_human[..., 2] < 0.2] = 0
        relation_dog[relation_dog[..., 2] < 0.2] = 0
        relation = extract_relation_features(interpolate_missing(relation_human),
                                             interpolate_missing(relation_dog), boxes, image_size)
        human = normalize_human_skeleton(human, image_size)
        dog = normalize_dog_skeleton(dog, image_size)
        if self.training and self.strong_augmentation and len(human) > self.target_frames:
            # Random sub-window changes both temporal crop and apparent speed.
            span = rng.uniform(0.75, 1.0) * (len(human) - 1)
            start = rng.uniform(0, max((len(human) - 1) - span, 0))
            positions = np.linspace(start, start + span, self.target_frames)
            step = span / max(self.target_frames - 1, 1)
            positions += rng.uniform(-0.25 * step, 0.25 * step, self.target_frames)
            positions[0], positions[-1] = start, start + span
            indices = np.rint(np.maximum.accumulate(
                np.clip(positions, 0, len(human) - 1)
            )).astype(np.int64)
            mask = np.ones(self.target_frames, dtype=np.float32)
        else:
            indices, mask = temporal_indices(len(human), self.target_frames,
                                             training=self.training, rng=rng)
        human = apply_temporal_indices(human, indices, mask, self.target_frames)
        dog = apply_temporal_indices(dog, indices, mask, self.target_frames)
        relation = apply_temporal_indices(relation, indices, mask, self.target_frames)
        def tensor(points):
            return torch.from_numpy(np.ascontiguousarray(points.transpose(2, 0, 1)[..., None]))
        return {
            "human_keypoints": tensor(human), "dog_keypoints": tensor(dog),
            "relation_features": torch.from_numpy(np.ascontiguousarray(relation)),
            "frame_mask": torch.from_numpy(mask),
            "label": torch.tensor(CLASS_TO_INDEX[row["interaction"]], dtype=torch.long),
            "clip_id": row["clip_id"], "dog_id": row["dog_id"], "person_id": row["person_id"],
        }


def class_counts(rows):
    counts = np.zeros(len(INTERACTION_CLASSES), dtype=np.int64)
    for row in rows: counts[CLASS_TO_INDEX[row["interaction"]]] += 1
    return counts


def summarize_relation_features(features: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Convert a relation sequence into robust statistics and feeding phases."""
    values = np.asarray(features, dtype=np.float32)
    if mask is not None:
        values = values[np.asarray(mask) > 0]
    if not len(values):
        return np.zeros(features.shape[-1] * 7 + 16, dtype=np.float32)

    statistics = np.concatenate((values.mean(0), values.std(0), values.min(0), values.max(0),
                                 np.percentile(values, (25, 50, 75), axis=0).reshape(-1)))

    # A feeding event is temporal: one hand approaches the mouth, remains near
    # it, and often withdraws. Percentiles alone lose that ordering, so retain
    # a compact set of phase/duration descriptors. Indices 8/9/10/11 are the
    # nearest-hand distance/closing speed and individual wrist distances;
    # index 21 is wrist-to-trunk distance.
    hand_distance = values[:, 8]
    hand_closing = values[:, 9]
    left_distance, right_distance = values[:, 10], values[:, 11]
    trunk_distance = values[:, 21]

    def longest_run(condition):
        best = current = 0
        for active in condition:
            current = current + 1 if active else 0
            best = max(best, current)
        return best / max(len(condition), 1)

    contact = hand_distance < 0.25
    near = hand_distance < 0.35
    stable_contact = contact & (np.abs(hand_closing) < 0.025)
    section = max(1, len(values) // 4)
    start_distance = float(hand_distance[:section].mean())
    middle_minimum = float(hand_distance.min())
    end_distance = float(hand_distance[-section:].mean())
    approach_drop = max(start_distance - middle_minimum, 0.0)
    withdrawal_rise = max(end_distance - middle_minimum, 0.0)
    closer_to_mouth = hand_distance < trunk_distance
    selected_hand = (right_distance < left_distance).astype(np.int8)
    switch_rate = (float(np.mean(selected_hand[1:] != selected_hand[:-1]))
                   if len(selected_hand) > 1 else 0.0)
    dominant_hand_ratio = max(float(selected_hand.mean()), 1.0 - float(selected_hand.mean()))
    longest_contact = longest_run(contact)
    feeding_phases = np.asarray((
        longest_contact, longest_run(near), float(stable_contact.mean()),
        start_distance, middle_minimum, end_distance,
        approach_drop, withdrawal_rise,
        float((trunk_distance - hand_distance).mean()),
        float(closer_to_mouth.mean()),
        float((contact & closer_to_mouth).mean()),
        dominant_hand_ratio, switch_rate,
        float(np.percentile(hand_closing, 90)),
        float(np.percentile(-hand_closing, 90)),
        approach_drop * longest_contact,
    ), dtype=np.float32)
    return np.concatenate((statistics, feeding_phases)).astype(np.float32)
