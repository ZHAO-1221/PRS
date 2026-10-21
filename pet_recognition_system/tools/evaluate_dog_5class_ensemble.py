"""One-shot evaluation of the frozen five-class dog-behaviour ensemble."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

from cross_validate_dog_joint_specialist import combined_features
from pet_recognition.datasets.dog_behavior import DOG_BEHAVIOR_CLASSES, DogBehaviorSkeletonDataset
from pet_recognition.evaluation.classification_metrics import classification_metrics, save_classification_report
from train_dog_play_specialist import tcn_probabilities
from train_dog_eating_specialist import dataset_features as eating_features


INDEX = {name: index for index, name in enumerate(DOG_BEHAVIOR_CLASSES)}


def rgb_features(dataset, feature_dir, mode):
    rows = []
    for row in dataset.rows:
        with np.load(Path(feature_dir) / f"{row['clip_id']}.npz", allow_pickle=False) as archive:
            if mode == "prompt":
                rows.append(archive["features"].astype(np.float32))
            else:
                values = archive["embeddings"].astype(np.float32)
                rows.append(np.concatenate((values.mean(0), values.std(0),
                                            np.abs(values[-1] - values[0]))))
    return np.stack(rows)


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def apply_pair(logits, features, artifact):
    probability = artifact["model"].predict_proba(features)[:, 1]
    decision = np.log((probability + 0.02) / (1 - probability + 0.02))
    positive, negative = INDEX[artifact["positive"]], INDEX[artifact["negative"]]
    if artifact["gate"] == "pair_top":
        ranking = np.argsort(logits, axis=1)[:, -2:]
        enabled = np.logical_and((ranking == positive).any(1),
                                 (ranking == negative).any(1))
    elif artifact["gate"] == "close":
        enabled = np.abs(logits[:, positive] - logits[:, negative]) <= artifact["margin"]
    else:
        enabled = np.ones(len(logits), dtype=bool)
    logits[enabled, positive] += artifact["alpha"] * decision[enabled]
    logits[enabled, negative] -= artifact["alpha"] * decision[enabled]
    return logits


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(
        project / "outputs" / "dog_behavior_manifest_5class_v4.csv"))
    parser.add_argument("--skeleton-dir", required=True)
    parser.add_argument("--cv-dir", default=str(
        project / "work_dirs" / "dog_behavior_5class_v4_cv"))
    parser.add_argument("--fusion-dir", default=str(
        project / "work_dirs" / "dog_behavior_5class_v4_fusion"))
    parser.add_argument("--output-dir", default=str(
        project / "work_dirs" / "dog_behavior_5class_v4_test"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rgb-fusion-dir")
    parser.add_argument("--prompt-dir")
    parser.add_argument("--raw-rgb-dir")
    parser.add_argument("--eating-context-fusion-dir")
    parser.add_argument("--eating-context-dir")
    parser.add_argument("--feature-version", default="fixed_region_v2",
                        choices=("head_duration_v1", "fixed_region_v2"))
    args = parser.parse_args()

    dataset = DogBehaviorSkeletonDataset(
        manifest=args.manifest, skeleton_dir=args.skeleton_dir, split="test",
        target_frames=32, training=False, augment=False)
    features, labels = combined_features(dataset, args.feature_version)
    eating_x, eating_labels = eating_features(
        dataset, include_fixed_region=args.feature_version == "fixed_region_v2")
    if not np.array_equal(labels, eating_labels):
        raise AssertionError("eating feature order mismatch")
    cv_dir, fusion_dir = Path(args.cv_dir), Path(args.fusion_dir)
    rgb_dir = Path(args.rgb_fusion_dir) if args.rgb_fusion_dir else None
    fold_probabilities = []
    for fold_index in range(args.folds):
        probability = tcn_probabilities(
            cv_dir / f"fold_{fold_index}" / "best.pt", dataset, args.device)
        logits = np.log(probability + 1e-8)
        multiclass = joblib.load(
            fusion_dir / f"fold_{fold_index}_multiclass.joblib")
        logits += multiclass["alpha"] * np.log(
            multiclass["model"].predict_proba(features) + 1e-5)
        for stage_index in range(1, args.stages + 1):
            artifact = joblib.load(
                fusion_dir / f"stage_{stage_index}_fold_{fold_index}.joblib")
            logits = apply_pair(logits, features, artifact)
        duration_path = fusion_dir / f"fold_{fold_index}_eating_duration.joblib"
        if duration_path.exists():
            artifact = joblib.load(duration_path)
            probability = artifact["model"].predict_proba(eating_x)[:, 1]
            decision = np.log((probability + 0.02) / (1 - probability + 0.02))
            eating_index = INDEX["eating_drinking"]
            if artifact["gate"] == "close":
                other = np.max(np.delete(logits, eating_index, axis=1), axis=1)
                enabled = np.abs(logits[:, eating_index] - other) <= artifact["margin"]
            else:
                enabled = np.ones(len(logits), dtype=bool)
            logits[enabled, eating_index] += artifact["alpha"] * decision[enabled]
        if rgb_dir is not None:
            rgb_artifact = joblib.load(rgb_dir / f"fold_{fold_index}_rgb.joblib")
            feature_root = args.prompt_dir if rgb_artifact["mode"] == "prompt" else args.raw_rgb_dir
            if not feature_root:
                parser.error(f"RGB artifact requires --{rgb_artifact['mode']}-dir")
            visual = rgb_features(dataset, feature_root, rgb_artifact["mode"])
            logits += rgb_artifact["alpha"] * np.log(
                rgb_artifact["model"].predict_proba(visual) + 1e-5)
        if args.eating_context_fusion_dir:
            artifact = joblib.load(Path(args.eating_context_fusion_dir) /
                                   f"fold_{fold_index}_eating_context.joblib")
            if not args.eating_context_dir:
                parser.error("eating context fusion requires --eating-context-dir")
            context = []
            for row in dataset.rows:
                with np.load(Path(args.eating_context_dir) /
                             f"{row['clip_id']}.npz", allow_pickle=False) as archive:
                    context.append(archive["features"].astype(np.float32))
            probability = artifact["model"].predict_proba(np.stack(context))[:, 1]
            decision = np.log((probability + 0.02) / (1 - probability + 0.02))
            eating_index = INDEX["eating_drinking"]
            if artifact["gate"] == "close":
                other = np.max(np.delete(logits, eating_index, axis=1), axis=1)
                enabled = np.abs(logits[:, eating_index] - other) <= artifact["margin"]
            else:
                enabled = np.ones(len(logits), dtype=bool)
            logits[enabled, eating_index] += artifact["alpha"] * decision[enabled]
        fold_probabilities.append(softmax(logits))

    probability = np.mean(fold_probabilities, axis=0)
    prediction = probability.argmax(1)
    result = classification_metrics(labels, prediction, DOG_BEHAVIOR_CLASSES)
    output = Path(args.output_dir)
    save_classification_report(result, DOG_BEHAVIOR_CLASSES, output, "test")
    with (output / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("clip_id", "dog_id", "actual", "predicted", "confidence"))
        for row, actual, predicted, values in zip(dataset.rows, labels, prediction, probability):
            writer.writerow((row["clip_id"], row["dog_id"],
                             DOG_BEHAVIOR_CLASSES[actual],
                             DOG_BEHAVIOR_CLASSES[predicted],
                             f"{values[predicted]:.6f}"))
    print({"accuracy": result["accuracy"], "macro_f1": result["macro_f1"],
           "correct": int((labels == prediction).sum()), "total": len(labels)})


if __name__ == "__main__":
    main()
