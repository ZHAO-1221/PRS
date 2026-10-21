"""Fit and fuse a motion-statistics specialist for ``playing_alone``."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier

from pet_recognition.datasets.dog_behavior import (
    DOG_BEHAVIOR_CLASSES, DogBehaviorSkeletonDataset,
)
from pet_recognition.evaluation.classification_metrics import (
    classification_metrics, save_classification_report,
)
from pet_recognition.models.skeleton_tcn import (
    DogBehaviorMotionAuxTCN,
    DogBehaviorMotionSummaryTCN,
    DogBehaviorTCN,
)


PLAY_INDEX = (DOG_BEHAVIOR_CLASSES.index("playing_alone")
              if "playing_alone" in DOG_BEHAVIOR_CLASSES else None)


def motion_statistics(item):
    """Summarize body translation and articulated motion for one clip."""
    skeleton = item["keypoints"].numpy()[..., 0].transpose(1, 2, 0)
    global_motion = item["global_motion"].numpy()
    mask = item["frame_mask"].numpy() > 0
    skeleton, global_motion = skeleton[mask], global_motion[mask]
    joint_velocity = np.diff(skeleton[..., :2], axis=0, prepend=skeleton[:1, ..., :2])
    joint_speed = np.linalg.norm(joint_velocity, axis=2)
    displacement = np.linalg.norm(global_motion[:, :2], axis=1)
    centre_speed = np.linalg.norm(global_motion[:, 2:4], axis=1)
    signals = np.column_stack((
        global_motion, displacement, centre_speed,
        joint_speed.mean(1), joint_speed.max(1),
    ))
    statistics = np.concatenate((
        signals.mean(0), signals.std(0), signals.min(0), signals.max(0),
        np.percentile(signals, (25, 50, 75), axis=0).reshape(-1),
    ))
    extras = np.asarray((
        centre_speed.sum(), joint_speed.mean(1).sum(),
        displacement[-1], np.ptp(global_motion[:, 0]), np.ptp(global_motion[:, 1]),
    ), dtype=np.float32)
    # Preserve *where* movement occurs. Playing with a toy may keep the dog's
    # body almost stationary while the nose and front paws move repeatedly.
    # Per-joint descriptors retain this signal that a whole-body mean loses.
    joint_range = np.linalg.norm(np.ptp(skeleton[..., :2], axis=0), axis=1)
    joint_descriptors = np.concatenate((
        joint_speed.mean(0), joint_speed.max(0),
        np.percentile(joint_speed, 90, axis=0), joint_range,
        skeleton[..., 2].mean(0),
    ))
    return np.concatenate((statistics, extras, joint_descriptors)).astype(np.float32)


def dataset_features(dataset):
    rows, labels = [], []
    for item in dataset:
        rows.append(motion_statistics(item)); labels.append(item["label"].item())
    return np.stack(rows), np.asarray(labels)


def tcn_probabilities(checkpoint, dataset, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model_name = saved.get("args", {}).get("model", "tcn")
    models = {
        "tcn": DogBehaviorTCN,
        "motion_tcn": DogBehaviorMotionSummaryTCN,
        "motion_aux_tcn": DogBehaviorMotionAuxTCN,
    }
    if model_name not in models:
        raise ValueError(f"unsupported specialist base model {model_name!r}")
    model = models[model_name](num_classes=len(saved.get("classes", DOG_BEHAVIOR_CLASSES))).to(device)
    model.load_state_dict(saved["model"]); model.eval()
    probabilities = []
    with torch.no_grad():
        for item in dataset:
            logits = model(item["keypoints"][None].to(device),
                           item["frame_mask"][None].to(device),
                           item["global_motion"][None].to(device))
            probabilities.append(torch.softmax(logits, 1).cpu().numpy()[0])
    return np.stack(probabilities)


def main():
    if PLAY_INDEX is None:
        raise RuntimeError("playing_alone is archived and is no longer an active dog-behaviour class")
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--manifest", default=str(project / "outputs" / "dog_behavior_manifest_v2.csv"))
    parser.add_argument("--data-root", default=os.environ.get("PET_DATA_ROOT"))
    parser.add_argument("--skeleton-dir")
    parser.add_argument("--output-dir", default=str(project / "work_dirs" / "dog_behavior_v2_play_specialist"))
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    base_prediction = base_probability.argmax(1)
    base_report = classification_metrics(val_labels, base_prediction, DOG_BEHAVIOR_CLASSES)

    best = None
    for depth in (2, 3, 4, None):
        for leaf in (1, 2, 3):
            for max_features in ("sqrt", 0.5):
                specialist = ExtraTreesClassifier(
                    n_estimators=500, max_depth=depth, min_samples_leaf=leaf,
                    max_features=max_features, class_weight="balanced",
                    random_state=args.seed,
                ).fit(train_x, train_labels == PLAY_INDEX)
                play_probability = specialist.predict_proba(val_x)[:, 1]
                play_log_odds = np.log((play_probability + 0.02) /
                                       (1 - play_probability + 0.02))
                for alpha in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
                    fused_logits = np.log(base_probability + 1e-8)
                    fused_logits[:, PLAY_INDEX] += alpha * play_log_odds
                    prediction = fused_logits.argmax(1)
                    report = classification_metrics(val_labels, prediction,
                                                    DOG_BEHAVIOR_CLASSES)
                    play_f1 = report["per_class"]["playing_alone"]["f1"]
                    # Do not buy play recall by materially degrading the other
                    # five behaviours on the validation set.
                    eligible = report["macro_f1"] >= base_report["macro_f1"] - 0.03
                    key = (eligible, play_f1, report["macro_f1"], report["accuracy"])
                    if best is None or key > best[0]:
                        best = (key, specialist, alpha, depth, leaf, max_features,
                                play_probability, prediction, report)

    _, specialist, alpha, depth, leaf, max_features, play_probability, prediction, report = best
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": specialist, "alpha": alpha, "play_index": PLAY_INDEX,
                 "classes": DOG_BEHAVIOR_CLASSES, "base_checkpoint": str(args.checkpoint),
                 "feature_dim": train_x.shape[1]}, output / "play_specialist.joblib")
    save_classification_report(report, DOG_BEHAVIOR_CLASSES, output, "val_fused")
    summary = {
        "base_accuracy": base_report["accuracy"],
        "base_macro_f1": base_report["macro_f1"],
        "base_playing_alone": base_report["per_class"]["playing_alone"],
        "fused_accuracy": report["accuracy"],
        "fused_macro_f1": report["macro_f1"],
        "fused_playing_alone": report["per_class"]["playing_alone"],
        "specialist": {"max_depth": depth, "min_samples_leaf": leaf,
                       "max_features": max_features, "alpha": alpha},
    }
    (output / "val_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
