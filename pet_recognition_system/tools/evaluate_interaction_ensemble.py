"""Evaluate a frozen fold ensemble by averaging class probabilities."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import joblib
import numpy as np

from pet_recognition.datasets.interaction import (
    INTERACTION_CLASSES, InteractionSkeletonDataset, summarize_relation_features,
)
from pet_recognition.evaluation.classification_metrics import (
    classification_metrics, save_classification_report,
)


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--data-root", default=os.environ.get("PET_DATA_ROOT"))
    parser.add_argument("--skeleton-dir")
    parser.add_argument("--manifest", default=str(project / "outputs" / "interaction_manifest.csv"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", help="write reports here instead of beside the weights")
    args = parser.parse_args()
    if not args.data_root and not args.skeleton_dir:
        parser.error("set PET_DATA_ROOT/--data-root or pass --skeleton-dir")
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir
    model_paths = sorted(model_dir.glob("fold_*.joblib"))
    if not model_paths:
        raise FileNotFoundError(f"no fold_*.joblib models in {model_dir}")
    models = [joblib.load(path)["model"] for path in model_paths]
    skeleton_dir = (Path(args.skeleton_dir) if args.skeleton_dir
                    else Path(args.data_root) / "skeletons_generated")
    dataset = InteractionSkeletonDataset(
        args.manifest, skeleton_dir, args.split, training=False
    )
    feature_rows, targets = [], []
    for item in dataset:
        feature_rows.append(summarize_relation_features(
            item["relation_features"].numpy(), item["frame_mask"].numpy()
        ))
        targets.append(item["label"].item())
    features = np.stack(feature_rows)
    probabilities = np.mean([model.predict_proba(features) for model in models], axis=0)
    predictions = probabilities.argmax(axis=1)
    report = classification_metrics(targets, predictions, INTERACTION_CLASSES)
    prefix = f"ensemble_{args.split}"
    save_classification_report(report, INTERACTION_CLASSES, output_dir, prefix)
    with (output_dir / f"{prefix}_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("clip_id", "dog_id", "person_id", "actual", "predicted",
                         *[f"prob_{name}" for name in INTERACTION_CLASSES]))
        for row, target, prediction, probability in zip(
            dataset.rows, targets, predictions, probabilities
        ):
            writer.writerow((row["clip_id"], row["dog_id"], row["person_id"],
                             INTERACTION_CLASSES[target], INTERACTION_CLASSES[prediction],
                             *probability.tolist()))
    print(json.dumps({"models": len(models), **report}, indent=2))


if __name__ == "__main__":
    main()
