"""Skeleton cleaning, normalization and temporal sampling."""
from __future__ import annotations

import numpy as np


def interpolate_missing(keypoints: np.ndarray) -> np.ndarray:
    """Linearly fill missing x/y values while keeping their confidence at zero."""
    output = np.asarray(keypoints, dtype=np.float32).copy()
    timeline = np.arange(output.shape[0], dtype=np.float32)
    for joint in range(output.shape[1]):
        observed = output[:, joint, 2] > 0
        if not observed.any():
            continue
        for axis in (0, 1):
            output[:, joint, axis] = np.interp(
                timeline, timeline[observed], output[observed, joint, axis]
            )
    return output


def normalize_dog_skeleton(keypoints: np.ndarray, image_size: np.ndarray) -> np.ndarray:
    """Make coordinates translation/scale invariant on every frame."""
    points = np.asarray(keypoints, dtype=np.float32).copy()
    width, height = np.asarray(image_size, dtype=np.float32)
    minimum_scale = max(float(width), float(height)) * 0.02
    for frame in range(points.shape[0]):
        visible = points[frame, :, 2] > 0
        available = visible | np.any(points[frame, :, :2] != 0, axis=1)
        if not available.any():
            points[frame] = 0
            continue
        if visible[3] and visible[4]:
            center = (points[frame, 3, :2] + points[frame, 4, :2]) / 2
        elif visible.any():
            center = points[frame, visible, :2].mean(axis=0)
        else:
            center = points[frame, available, :2].mean(axis=0)
        scale_points = points[frame, visible if visible.any() else available, :2]
        xy = points[frame, available, :2]
        extent = max(float(np.ptp(scale_points[:, 0])), float(np.ptp(scale_points[:, 1])), minimum_scale)
        points[frame, available, :2] = (xy - center) / extent
        points[frame, ~available, :2] = 0
    return points


def normalize_human_skeleton(keypoints: np.ndarray, image_size: np.ndarray) -> np.ndarray:
    """Center and scale COCO-17 human joints on every frame."""
    points = np.asarray(keypoints, dtype=np.float32).copy()
    width, height = np.asarray(image_size, dtype=np.float32)
    minimum_scale = max(float(width), float(height)) * 0.02
    for frame in range(points.shape[0]):
        visible = points[frame, :, 2] > 0
        available = visible | np.any(points[frame, :, :2] != 0, axis=1)
        if not available.any():
            points[frame] = 0; continue
        if available[11] and available[12]:
            center = (points[frame, 11, :2] + points[frame, 12, :2]) / 2
        elif available[5] and available[6]:
            center = (points[frame, 5, :2] + points[frame, 6, :2]) / 2
        else:
            center = points[frame, available, :2].mean(axis=0)
        xy = points[frame, available, :2]
        extent = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), minimum_scale)
        points[frame, available, :2] = (xy - center) / extent
        points[frame, ~available, :2] = 0
    return points


def extract_relation_features(human: np.ndarray, dog: np.ndarray, boxes: np.ndarray,
                              image_size: np.ndarray) -> np.ndarray:
    """Return 48 frame-level interaction features.

    Besides global geometry and hand-to-mouth cues, the final 15 channels
    describe sustained hand/arm contact with the dog's trunk, hugging and
    relative stillness.  The last 12 channels describe single- and dual-subject
    large/rapid motion for playing together. Clip-level duration is represented
    later by summary statistics over these frame-level scores.
    """
    width, height = np.asarray(image_size, dtype=np.float32)
    scale = np.array([width, height], dtype=np.float32)
    # Relative displacement is reconstructed from each trajectory plus its
    # first valid detector centre, which preserves cross-subject geometry.
    def box_centers(stream, normalize=True):
        b = np.asarray(stream, np.float32)
        c = np.column_stack(((b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2))
        valid = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
        timeline = np.arange(len(b))
        if valid.any():
            for axis in range(2): c[:, axis] = np.interp(timeline, timeline[valid], c[valid, axis])
        else: c.fill(0)
        return c / scale if normalize else c
    hc, dc = box_centers(boxes[0]), box_centers(boxes[1])
    relative = dc - hc
    relative_velocity = np.vstack((np.zeros((1, 2), np.float32), np.diff(relative, axis=0)))
    distance = np.linalg.norm(relative, axis=1, keepdims=True)
    closing = -np.vstack((np.zeros((1, 1), np.float32), np.diff(distance, axis=0)))

    # Feeding is characterized by a wrist approaching and remaining near the
    # dog's mouth. AP-10K has no mouth joint, so its nose (joint 2) is the
    # closest stable proxy. Normalize by dog-box diagonal, not image size, so
    # the same physical contact has a comparable magnitude in near/far views.
    dog_width = boxes[1, :, 2] - boxes[1, :, 0]
    dog_height = boxes[1, :, 3] - boxes[1, :, 1]
    valid_box = (dog_width > 1) & (dog_height > 1)
    dog_diagonal = np.full(len(dog_width), np.nan, dtype=np.float32)
    dog_diagonal[valid_box] = np.sqrt(dog_width[valid_box] ** 2 + dog_height[valid_box] ** 2)
    timeline = np.arange(len(dog_width))
    if valid_box.any():
        dog_diagonal = np.interp(timeline, timeline[valid_box], dog_diagonal[valid_box])
    else:
        dog_diagonal.fill(max(width, height) * 0.2)
    dog_scale = np.maximum(dog_diagonal, 1.0)[:, None]
    hands = human[:, (9, 10), :2]
    mouth = dog[:, 2, :2]
    hand_vectors = np.clip((mouth[:, None, :] - hands) / dog_scale[:, None, :], -3, 3)
    hand_distances = np.linalg.norm(hand_vectors, axis=2)
    nearest = hand_distances.argmin(axis=1)
    hand_vector = hand_vectors[np.arange(len(human)), nearest]
    hand_distance = hand_distances[np.arange(len(human)), nearest, None]
    hand_closing = -np.vstack((np.zeros((1, 1), np.float32), np.diff(hand_distance, axis=0)))
    contact_score = np.exp(-hand_distance / 0.20).astype(np.float32)
    contact_flag = (hand_distance < 0.25).astype(np.float32)
    nose_confidence = dog[:, 2, 2:3]
    wrist_confidence = human[np.arange(len(human)), np.asarray((9, 10))[nearest], 2:3]

    human_orientation = human[:, 0, :2] - human[:, (5, 6), :2].mean(axis=1)
    dog_orientation = dog[:, 2, :2] - dog[:, 3, :2]
    def unit(vector):
        return vector / np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-6)
    human_orientation, dog_orientation = unit(human_orientation), unit(dog_orientation)
    orientation_dot = (human_orientation * dog_orientation).sum(axis=1, keepdims=True)

    # Petting/hugging cues.  Distances are measured to the dog's trunk graph,
    # rather than to a single joint, because a hand may touch the head, back,
    # chest or flank.  Midpoints along both human arms make the distance a
    # useful approximation of arm/trunk intersection during a hug.
    trunk_edges = ((3, 5), (3, 8), (5, 11), (8, 14),
                   (11, 4), (14, 4), (3, 4))

    def distance_to_trunk(points):
        points = np.asarray(points, dtype=np.float32)
        point_valid = np.any(points != 0, axis=2)
        best = np.full(points.shape[:2], np.inf, dtype=np.float32)
        dog_xy = dog[..., :2]
        dog_valid = (dog[..., 2] > 0) | np.any(dog_xy != 0, axis=2)
        for start_index, end_index in trunk_edges:
            start = dog_xy[:, start_index][:, None, :]
            end = dog_xy[:, end_index][:, None, :]
            segment = end - start
            denominator = np.maximum((segment * segment).sum(2), 1e-6)
            projection = np.clip(((points - start) * segment).sum(2) / denominator, 0, 1)
            nearest_point = start + projection[..., None] * segment
            distance_to_edge = np.linalg.norm(points - nearest_point, axis=2)
            edge_valid = dog_valid[:, start_index] & dog_valid[:, end_index]
            distance_to_edge[~(point_valid & edge_valid[:, None])] = np.inf
            best = np.minimum(best, distance_to_edge)
        best = np.min(best, axis=1, keepdims=True)
        best[~np.isfinite(best)] = dog_scale[~np.isfinite(best)] * 3
        return np.clip(best / dog_scale, 0, 3).astype(np.float32)

    shoulders = human[:, (5, 6), :2]
    elbows = human[:, (7, 8), :2]
    wrists = human[:, (9, 10), :2]
    arm_points = np.concatenate((shoulders, elbows, wrists,
                                 (shoulders + elbows) / 2,
                                 (elbows + wrists) / 2), axis=1)
    torso_points = human[:, (5, 6, 11, 12), :2]
    wrist_trunk_distance = distance_to_trunk(wrists)
    elbow_trunk_distance = distance_to_trunk(elbows)
    arm_trunk_distance = distance_to_trunk(arm_points)
    torso_trunk_distance = distance_to_trunk(torso_points)
    trunk_contact_score = np.exp(-arm_trunk_distance / 0.20).astype(np.float32)
    trunk_contact_flag = (arm_trunk_distance < 0.25).astype(np.float32)
    arm_cross_score = np.exp(-arm_trunk_distance / 0.12).astype(np.float32)

    def joint_speed(points):
        movement = np.linalg.norm(np.diff(points, axis=0, prepend=points[:1]), axis=2)
        return (movement.mean(axis=1, keepdims=True) / dog_scale).astype(np.float32)

    wrist_speed = joint_speed(wrists)
    elbow_speed = joint_speed(elbows)
    relative_center_speed = np.linalg.norm(relative_velocity, axis=1, keepdims=True)
    hc_pixels, dc_pixels = box_centers(boxes[0], False), box_centers(boxes[1], False)
    human_center_speed = (np.linalg.norm(np.diff(hc_pixels, axis=0, prepend=hc_pixels[:1]),
                                         axis=1, keepdims=True) / dog_scale).astype(np.float32)
    dog_center_speed = (np.linalg.norm(np.diff(dc_pixels, axis=0, prepend=dc_pixels[:1]),
                                       axis=1, keepdims=True) / dog_scale).astype(np.float32)

    human_box, dog_box = boxes[0, :, :4], boxes[1, :, :4]
    intersection_width = np.maximum(0, np.minimum(human_box[:, 2], dog_box[:, 2]) -
                                    np.maximum(human_box[:, 0], dog_box[:, 0]))
    intersection_height = np.maximum(0, np.minimum(human_box[:, 3], dog_box[:, 3]) -
                                     np.maximum(human_box[:, 1], dog_box[:, 1]))
    dog_area = np.maximum((dog_box[:, 2] - dog_box[:, 0]) *
                          (dog_box[:, 3] - dog_box[:, 1]), 1)
    box_intersection = np.clip(intersection_width * intersection_height / dog_area,
                               0, 1)[:, None].astype(np.float32)
    stability_score = np.exp(-relative_center_speed / 0.025).astype(np.float32)
    hug_score = (arm_cross_score * stability_score *
                 np.sqrt(np.clip(box_intersection, 0, 1))).astype(np.float32)

    # Playing-together cues.  Keep both mean and maximum articulated velocity:
    # a running dog moves many joints, while a person waving or throwing may
    # move only one arm quickly.  Combining subject activity with max() supports
    # both "both active" and "one clearly active" cases from the label rule.
    human_width = boxes[0, :, 2] - boxes[0, :, 0]
    human_height = boxes[0, :, 3] - boxes[0, :, 1]
    human_diagonal = np.sqrt(np.maximum(human_width, 1) ** 2 +
                             np.maximum(human_height, 1) ** 2)[:, None]

    def skeleton_speed(points, subject_scale):
        xy = points[..., :2]
        available = (points[..., 2] > 0) | np.any(xy != 0, axis=2)
        displacement = np.linalg.norm(np.diff(xy, axis=0, prepend=xy[:1]), axis=2)
        valid_pair = available & np.vstack((available[:1], available[:-1]))
        displacement[~valid_pair] = np.nan
        mean_speed = np.nanmean(displacement, axis=1, keepdims=True)
        max_speed = np.nanmax(displacement, axis=1, keepdims=True)
        mean_speed = np.nan_to_num(mean_speed) / subject_scale
        max_speed = np.nan_to_num(max_speed) / subject_scale
        return np.clip(mean_speed, 0, 3).astype(np.float32), np.clip(max_speed, 0, 3).astype(np.float32)

    human_pose_speed, human_pose_max_speed = skeleton_speed(human, human_diagonal)
    dog_pose_speed, dog_pose_max_speed = skeleton_speed(dog, dog_scale)
    human_activity = human_center_speed + human_pose_speed
    dog_activity = dog_center_speed + dog_pose_speed
    either_activity = np.maximum(human_activity, dog_activity)
    both_activity = np.sqrt(np.maximum(human_activity * dog_activity, 0))
    activity_difference = np.abs(human_activity - dog_activity)
    human_acceleration = np.abs(np.diff(human_activity, axis=0, prepend=human_activity[:1]))
    dog_acceleration = np.abs(np.diff(dog_activity, axis=0, prepend=dog_activity[:1]))
    play_motion_score = either_activity + 0.5 * np.minimum(human_activity, dog_activity)

    return np.concatenate((relative, distance, relative_velocity, closing, hand_vector,
                           hand_distance, hand_closing, hand_distances, contact_score,
                           contact_flag, human_orientation, dog_orientation,
                           orientation_dot, nose_confidence, wrist_confidence,
                           wrist_trunk_distance, elbow_trunk_distance,
                           arm_trunk_distance, torso_trunk_distance,
                           trunk_contact_score, trunk_contact_flag, arm_cross_score,
                           wrist_speed, elbow_speed, relative_center_speed,
                           human_center_speed, dog_center_speed, box_intersection,
                           stability_score, hug_score,
                           human_pose_speed, human_pose_max_speed,
                           dog_pose_speed, dog_pose_max_speed,
                           human_activity, dog_activity, either_activity,
                           both_activity, activity_difference,
                           human_acceleration, dog_acceleration,
                           play_motion_score), axis=1).astype(np.float32)


def _motion_features(centers: np.ndarray, scales: np.ndarray, image_size: np.ndarray) -> np.ndarray:
    """Convert possibly incomplete image-space centres/scales into six features."""
    width, height = np.asarray(image_size, dtype=np.float32)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size {image_size}")
    centers = np.asarray(centers, dtype=np.float32).copy()
    scales = np.asarray(scales, dtype=np.float32).copy()
    total = centers.shape[0]
    valid = np.isfinite(centers[:, 0])
    if not valid.any():
        return np.zeros((total, 6), dtype=np.float32)
    timeline = np.arange(total, dtype=np.float32)
    for axis in range(2):
        centers[:, axis] = np.interp(timeline, timeline[valid], centers[valid, axis])
    scales = np.interp(timeline, timeline[valid], scales[valid])

    # Detector boxes and pose estimates jitter slightly even for a stationary
    # dog. A short edge-padded moving average removes that noise without
    # suppressing clip-level translation.
    if total >= 3:
        window = min(5, total if total % 2 else total - 1)
        pad = window // 2
        kernel = np.ones(window, dtype=np.float32) / window
        for axis in range(2):
            centers[:, axis] = np.convolve(
                np.pad(centers[:, axis], (pad, pad), mode="edge"), kernel, mode="valid"
            )
        scales = np.convolve(np.pad(scales, (pad, pad), mode="edge"), kernel, mode="valid")

    centers /= np.array([width, height], dtype=np.float32)
    displacement = centers - centers[0]
    velocity = np.vstack((np.zeros((1, 2), dtype=np.float32), np.diff(centers, axis=0)))
    log_scale = np.log(np.maximum(scales, 1.0) / max(float(scales[0]), 1.0))[:, None]
    scale_velocity = np.vstack((np.zeros((1, 1), dtype=np.float32), np.diff(log_scale, axis=0)))
    return np.concatenate((displacement, velocity, log_scale, scale_velocity), axis=1).astype(np.float32)


def extract_global_motion(keypoints: np.ndarray, image_size: np.ndarray) -> np.ndarray:
    """Estimate global motion from keypoints (fallback when boxes are absent)."""
    points = np.asarray(keypoints, dtype=np.float32)
    total = points.shape[0]
    centers = np.full((total, 2), np.nan, dtype=np.float32)
    scales = np.full(total, np.nan, dtype=np.float32)
    for frame in range(total):
        available = (points[frame, :, 2] > 0) | np.any(points[frame, :, :2] != 0, axis=1)
        if not available.any():
            continue
        if available[3] and available[4]:
            centers[frame] = (points[frame, 3, :2] + points[frame, 4, :2]) / 2
        else:
            centers[frame] = points[frame, available, :2].mean(axis=0)
        xy = points[frame, available, :2]
        scales[frame] = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1.0)
    return _motion_features(centers, scales, image_size)


def extract_box_motion(boxes: np.ndarray, image_size: np.ndarray) -> np.ndarray:
    """Estimate global motion from detector boxes in ``x1,y1,x2,y2,score`` form."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] < 4:
        raise ValueError(f"expected boxes (T,>=4), got {boxes.shape}")
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    valid = (widths > 0) & (heights > 0)
    if boxes.shape[1] >= 5:
        valid &= boxes[:, 4] > 0
    centers = np.full((boxes.shape[0], 2), np.nan, dtype=np.float32)
    scales = np.full(boxes.shape[0], np.nan, dtype=np.float32)
    centers[valid, 0] = (boxes[valid, 0] + boxes[valid, 2]) / 2
    centers[valid, 1] = (boxes[valid, 1] + boxes[valid, 3]) / 2
    scales[valid] = np.sqrt(widths[valid] * heights[valid])
    return _motion_features(centers, scales, image_size)


def temporal_indices(
    total: int,
    target_frames: int = 32,
    *,
    training: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one set of indices that can be shared by every input stream."""
    if total <= 0:
        raise ValueError("a sequence must contain at least one frame")
    if total < target_frames:
        indices = np.arange(total, dtype=np.int64)
        mask = np.zeros(target_frames, dtype=np.float32)
        mask[:total] = 1
        return indices, mask
    positions = np.linspace(0, total - 1, target_frames)
    if training and total > target_frames:
        rng = rng or np.random.default_rng()
        step = (total - 1) / (target_frames - 1)
        positions += rng.uniform(-0.35 * step, 0.35 * step, size=target_frames)
        positions[0], positions[-1] = 0, total - 1
        positions = np.maximum.accumulate(np.clip(positions, 0, total - 1))
    return np.rint(positions).astype(np.int64), np.ones(target_frames, dtype=np.float32)


def apply_temporal_indices(
    sequence: np.ndarray, indices: np.ndarray, mask: np.ndarray, target_frames: int
) -> np.ndarray:
    """Sample or pad one stream using precomputed shared temporal indices."""
    sequence = np.asarray(sequence, dtype=np.float32)
    selected = sequence[indices]
    if selected.shape[0] == target_frames:
        return selected
    output = np.zeros((target_frames, *sequence.shape[1:]), dtype=np.float32)
    output[: selected.shape[0]] = selected
    return output


def temporal_sample(
    keypoints: np.ndarray,
    target_frames: int = 32,
    *,
    training: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample long clips and zero-pad short clips."""
    sequence = np.asarray(keypoints, dtype=np.float32)
    indices, mask = temporal_indices(
        sequence.shape[0], target_frames, training=training, rng=rng
    )
    return apply_temporal_indices(sequence, indices, mask, target_frames), mask


def augment_skeleton(
    keypoints: np.ndarray,
    rng: np.random.Generator,
    global_motion: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Apply small geometry noise and confidence-aware joint dropout."""
    output = np.asarray(keypoints, dtype=np.float32).copy()
    angle = rng.uniform(-10, 10) * np.pi / 180
    scale = rng.uniform(0.9, 1.1)
    rotation = scale * np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32
    )
    visible = output[..., 2] > 0
    output[..., :2] = output[..., :2] @ rotation.T
    output[..., :2] += rng.normal(0, 0.01, size=output[..., :2].shape).astype(np.float32)
    output[..., :2][~visible] = 0
    if global_motion is not None:
        motion = np.asarray(global_motion, dtype=np.float32).copy()
        # Displacement and velocity are image-plane vectors, so they receive
        # the same rotation/scale as the local skeleton augmentation.
        motion[:, 0:2] = motion[:, 0:2] @ rotation.T
        motion[:, 2:4] = motion[:, 2:4] @ rotation.T
    if rng.random() < 0.3:
        joint = int(rng.integers(0, output.shape[1]))
        output[:, joint] = 0
    return (output, motion) if global_motion is not None else output
