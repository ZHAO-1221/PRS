"""Geometry and head-motion features for eating/drinking recognition."""
from __future__ import annotations

import numpy as np

from train_dog_play_specialist import motion_statistics


def _statistics(values: np.ndarray) -> np.ndarray:
    return np.concatenate((values.mean(0), values.std(0), values.min(0), values.max(0),
                           np.percentile(values, (25, 50, 75), axis=0).reshape(-1)))


def _longest_true_fraction(values: np.ndarray) -> float:
    """Longest consecutive true run, expressed as a fraction of the clip."""
    values = np.asarray(values, dtype=bool)
    if values.size == 0:
        return 0.0
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return float(np.max(edges[1::2] - edges[::2], initial=0) / values.size)


def eating_statistics(item, *, include_fixed_region: bool = True) -> np.ndarray:
    """Summarize sustained head-down geometry and small repetitive head motion."""
    skeleton = item["keypoints"].numpy()[..., 0].transpose(1, 2, 0)
    global_motion = item["global_motion"].numpy()
    mask = item["frame_mask"].numpy() > 0
    skeleton, global_motion = skeleton[mask], global_motion[mask]
    xy, confidence = skeleton[..., :2], skeleton[..., 2]

    nose, neck, tail = xy[:, 2], xy[:, 3], xy[:, 4]
    shoulders = xy[:, [5, 8]].mean(1)
    front_paws = xy[:, [7, 10]]
    rear_paws = xy[:, [13, 16]]
    all_paws = xy[:, [7, 10, 13, 16]]
    nose_velocity = np.diff(nose, axis=0, prepend=nose[:1])
    nose_speed = np.linalg.norm(nose_velocity, axis=1)
    neck_speed = np.linalg.norm(np.diff(neck, axis=0, prepend=neck[:1]), axis=1)
    centre_speed = np.linalg.norm(global_motion[:, 2:4], axis=1)

    nose_to_front_paw = np.linalg.norm(front_paws - nose[:, None], axis=2).min(1)
    nose_to_ground = all_paws[..., 1].mean(1) - nose[:, 1]
    head_drop = nose[:, 1] - neck[:, 1]
    head_below_shoulders = nose[:, 1] - shoulders[:, 1]
    trunk_length = np.linalg.norm(tail - neck, axis=1)
    safe_trunk = np.maximum(trunk_length, 1e-4)
    head_to_trunk = np.linalg.norm(nose - neck, axis=1) / np.maximum(trunk_length, 1e-4)
    paw_spread_y = all_paws[..., 1].max(1) - all_paws[..., 1].min(1)
    frame_features = np.column_stack((
        nose_to_front_paw,
        nose_to_ground,
        head_drop,
        head_below_shoulders,
        head_to_trunk,
        nose_speed,
        neck_speed,
        centre_speed,
        paw_spread_y,
        confidence[:, 2],
        confidence[:, 3],
        confidence[:, [7, 10]].mean(1),
    ))

    # Eating/drinking often produces repeated small vertical head movements.
    vertical = nose_velocity[:, 1]
    signs = np.sign(vertical)
    direction_changes = np.count_nonzero(signs[1:] * signs[:-1] < 0) / max(len(signs) - 1, 1)

    # Explicit duration features.  Coordinates are normalized per dog crop, so
    # thresholds are scaled by the neck-to-tail length instead of image pixels.
    drop_from_neck = head_drop / safe_trunk
    drop_from_shoulders = head_below_shoulders / safe_trunk
    low_neck = drop_from_neck > 0.08
    low_shoulders = drop_from_shoulders > 0.05
    strongly_low = np.logical_and(drop_from_neck > 0.15,
                                  drop_from_shoulders > 0.10)
    low_and_still = np.logical_and(
        low_neck,
        centre_speed <= np.percentile(centre_speed, 60),
    )
    nose_step = np.linalg.norm(nose_velocity, axis=1) / safe_trunk
    low_in_fixed_region = np.logical_and(low_neck, nose_step < 0.04)
    low_positions = nose[low_neck]
    if len(low_positions):
        low_position_radius = np.linalg.norm(
            low_positions - np.median(low_positions, axis=0), axis=1
        ) / np.maximum(np.median(safe_trunk[low_neck]), 1e-4)
        low_region_median_radius = float(np.median(low_position_radius))
        low_region_p90_radius = float(np.percentile(low_position_radius, 90))
    else:
        low_region_median_radius = low_region_p90_radius = 0.0
    duration_values = [
        low_neck.mean(),
        low_shoulders.mean(),
        strongly_low.mean(),
        low_and_still.mean(),
        _longest_true_fraction(low_neck),
        _longest_true_fraction(low_shoulders),
        _longest_true_fraction(strongly_low),
        _longest_true_fraction(low_and_still),
    ]
    if include_fixed_region:
        duration_values.extend((
            low_in_fixed_region.mean(),
            _longest_true_fraction(low_in_fixed_region),
            np.median(nose_step),
            np.percentile(nose_step, 75),
            low_region_median_radius,
            low_region_p90_radius,
        ))
    duration_values.extend((
        np.percentile(drop_from_neck, 25),
        np.percentile(drop_from_neck, 50),
        np.percentile(drop_from_neck, 75),
        np.percentile(drop_from_shoulders, 50),
    ))
    duration_features = np.asarray(duration_values, dtype=np.float32)
    extras = np.asarray((
        direction_changes,
        np.ptp(nose[:, 0]),
        np.ptp(nose[:, 1]),
        nose_speed.sum(),
        centre_speed.sum(),
        float(mask.sum()),
    ), dtype=np.float32)
    return np.concatenate((motion_statistics(item), _statistics(frame_features),
                           duration_features, extras)).astype(np.float32)


def dataset_features(dataset, *, include_fixed_region: bool = True):
    return (np.stack([eating_statistics(
                item, include_fixed_region=include_fixed_region) for item in dataset]),
            np.asarray([item["label"].item() for item in dataset]))
