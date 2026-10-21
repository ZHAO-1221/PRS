"""Fit a four-paws-ground, low-motion specialist for ``standing``."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from pet_recognition.datasets.dog_behavior import DOG_BEHAVIOR_CLASSES, DogBehaviorSkeletonDataset
from pet_recognition.evaluation.classification_metrics import (
    classification_metrics,
    save_classification_report,
)
from train_dog_play_specialist import motion_statistics, tcn_probabilities


STANDING_INDEX = DOG_BEHAVIOR_CLASSES.index("standing")


def _statistics(values: np.ndarray) -> np.ndarray:
    return np.concatenate((values.mean(0), values.std(0), values.min(0), values.max(0),
                           np.percentile(values, (25, 50, 75), axis=0).reshape(-1)))


def standing_statistics(item) -> np.ndarray:
    """Describe paw-ground alignment, leg extension, body orientation and stillness."""
    skeleton = item["keypoints"].numpy()[..., 0].transpose(1, 2, 0)
    motion = item["global_motion"].numpy()
    mask = item["frame_mask"].numpy() > 0
    skeleton, motion = skeleton[mask], motion[mask]
    xy, confidence = skeleton[..., :2], skeleton[..., 2]

    # AP-10K: front paws 7/10, rear paws 13/16; their proximal joints are
    # shoulder/elbow and hip/knee respectively.
    paw = xy[:, [7, 10, 13, 16]]
    paw_confidence = confidence[:, [7, 10, 13, 16]]
    proximal = xy[:, [5, 8, 11, 14]]
    middle = xy[:, [6, 9, 12, 15]]
    upper_leg = np.linalg.norm(middle - proximal, axis=2)
    lower_leg = np.linalg.norm(paw - middle, axis=2)
    direct_leg = np.linalg.norm(paw - proximal, axis=2)
    extension = direct_leg / np.maximum(upper_leg + lower_leg, 1e-4)

    paw_y = paw[..., 1]
    paw_ground_spread = paw_y.max(1) - paw_y.min(1)
    paw_below_body = paw_y.mean(1) - xy[:, [3, 4], 1].mean(1)
    body_vector = xy[:, 4] - xy[:, 3]
    body_horizontal = np.abs(body_vector[:, 0]) / np.maximum(
        np.linalg.norm(body_vector, axis=1), 1e-4
    )
    centre_speed = np.linalg.norm(motion[:, 2:4], axis=1)
    joint_speed = np.linalg.norm(
        np.diff(xy, axis=0, prepend=xy[:1]), axis=2
    )

    frame_features = np.column_stack((
        paw_ground_spread,
        paw_below_body,
        body_horizontal,
        extension.mean(1),
        extension.min(1),
        paw_confidence.mean(1),
        (paw_confidence > 0.2).sum(1) / 4.0,
        centre_speed,
        joint_speed.mean(1),
        joint_speed.max(1),
    ))
    return np.concatenate((motion_statistics(item), _statistics(frame_features))).astype(np.float32)


def dataset_features(dataset):
    return (np.stack([standing_statistics(item) for item in dataset]),
            np.asarray([item["label"].item() for item in dataset]))


def apply_binary_specialist(logits, features, artifact, class_index):
    probability = artifact["model"].predict_proba(features)[:, 1]
    log_odds = np.log((probability + 0.02) / (1 - probability + 0.02))
    logits[:, class_index] += artifact["alpha"] * log_odds
    return logits


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--play-specialist")
    parser.add_argument("--manifest", default=str(project / "outputs" / "dog_behavior_manifest_v2.csv"))
    parser.add_argument("--data-root", default=os.environ.get("PET_DATA_ROOT"))
    parser.add_argument("--skeleton-dir")
    parser.add_argument("--output-dir", default=str(project / "work_dirs" / "dog_behavior_v2_standing_specialist"))
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.data_root and not args.skeleton_dir:
        parser.error("set PET_DATA_ROOT/--data-root or pass --skeleton-dir")

    skeleton_dir = (Path(args.skeleton_dir) if args.skeleton_dir
                    else Path(args.data_root) / "skeletons_generated")
    common = dict(manifest=args.manifest, skeleton_dir=skeleton_dir,
                  target_frames=32, training=False, augment=False)
    train = DogBehaviorSkeletonDataset(split="train", **common)
    val = DogBehaviorSkeletonDataset(split="val", **common)
    train_x, train_labels = dataset_features(train)
    val_x, val_labels = dataset_features(val)
    base_probability = tcn_probabilities(args.checkpoint, val, args.device)
    base_logits = np.log(base_probability + 1e-8)

    if args.play_specialist:
        play_artifact = joblib.load(args.play_specialist)
        play_x = np.stack([motion_statistics(item) for item in val])
        base_logits = apply_binary_specialist(
            base_logits, play_x, play_artifact,
            DOG_BEHAVIOR_CLASSES.index("playing_alone"),
        )
    base_prediction = base_logits.argmax(1)
    base_report = classification_metrics(val_labels, base_prediction, DOG_BEHAVIOR_CLASSES)

    best = None
    for depth in (2, 3, 4, None):
        for leaf in (1, 2, 3, 4):
            for max_features in ("sqrt", 0.5, 0.75):
                specialist = ExtraTreesClassifier(
                    n_estimators=500, max_depth=depth, min_samples_leaf=leaf,
                    max_features=max_features, class_weight="balanced",
                    random_state=args.seed,
                ).fit(train_x, train_labels == STANDING_INDEX)
                probability = specialist.predict_proba(val_x)[:, 1]
                log_odds = np.log((probability + 0.02) / (1 - probability + 0.02))
                for alpha in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5):
                    logits = base_logits.copy()
                    logits[:, STANDING_INDEX] += alpha * log_odds
                    prediction = logits.argmax(1)
                    report = classification_metrics(val_labels, prediction, DOG_BEHAVIOR_CLASSES)
                    standing_f1 = report["per_class"]["standing"]["f1"]
                    eligible = report["macro_f1"] >= base_report["macro_f1"] - 0.02
                    key = (eligible, standing_f1, report["macro_f1"], report["accuracy"])
                    if best is None or key > best[0]:
                        best = (key, specialist, alpha, depth, leaf, max_features,
                                prediction, report)

    _, specialist, alpha, depth, leaf, max_features, prediction, report = best
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": specialist, "alpha": alpha, "class_index": STANDING_INDEX,
        "classes": DOG_BEHAVIOR_CLASSES, "base_checkpoint": str(args.checkpoint),
        "play_specialist": args.play_specialist, "feature_dim": train_x.shape[1],
    }, output / "standing_specialist.joblib")
    save_classification_report(report, DOG_BEHAVIOR_CLASSES, output, "val_fused")
    summary = {
        "base_accuracy": base_report["accuracy"],
        "base_macro_f1": base_report["macro_f1"],
        "base_standing": base_report["per_class"]["standing"],
        "fused_accuracy": report["accuracy"],
        "fused_macro_f1": report["macro_f1"],
        "fused_standing": report["per_class"]["standing"],
        "specialist": {"max_depth": depth, "min_samples_leaf": leaf,
                       "max_features": max_features, "alpha": alpha},
    }
    (output / "val_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
