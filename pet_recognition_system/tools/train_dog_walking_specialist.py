"""Trajectory persistence and gait features for walking/running recognition."""
from __future__ import annotations

import numpy as np

from train_dog_play_specialist import motion_statistics


def walking_statistics(item) -> np.ndarray:
    skeleton = item["keypoints"].numpy()[..., 0].transpose(1, 2, 0)
    motion = item["global_motion"].numpy()
    mask = item["frame_mask"].numpy() > 0
    skeleton, motion = skeleton[mask], motion[mask]
    velocity = motion[:, 2:4]
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.linalg.norm(np.diff(velocity, axis=0, prepend=velocity[:1]), axis=1)
    path = speed.sum()
    net = np.linalg.norm(motion[-1, :2])
    straightness = net / max(path, 1e-5)

    valid_direction = speed > 1e-4
    direction = np.arctan2(velocity[:, 1], velocity[:, 0])
    direction_change = np.abs(np.angle(np.exp(1j * np.diff(direction))))
    if valid_direction.sum() < 2:
        direction_change = np.zeros(1, dtype=np.float32)

    joint_velocity = np.diff(skeleton[..., :2], axis=0, prepend=skeleton[:1, ..., :2])
    paw_speed = np.linalg.norm(joint_velocity[:, [7, 10, 13, 16]], axis=2)
    paw_vertical = joint_velocity[:, [7, 10, 13, 16], 1]
    sign_changes = []
    for joint in range(4):
        signs = np.sign(paw_vertical[:, joint])
        sign_changes.append(np.count_nonzero(signs[1:] * signs[:-1] < 0) /
                            max(len(signs) - 1, 1))
    extras = np.asarray((
        path, net, straightness,
        speed.mean(), speed.std(), speed.max(),
        acceleration.mean(), acceleration.max(),
        np.mean(speed > 0.002), np.mean(speed > 0.005),
        direction_change.mean(), direction_change.std(), direction_change.max(),
        paw_speed.mean(), paw_speed.std(), paw_speed.max(),
        *sign_changes,
        np.abs(motion[:, 5]).mean(), np.abs(motion[:, 5]).max(),
    ), dtype=np.float32)
    return np.concatenate((motion_statistics(item), extras)).astype(np.float32)


def dataset_features(dataset):
    return (np.stack([walking_statistics(item) for item in dataset]),
            np.asarray([item["label"].item() for item in dataset]))
