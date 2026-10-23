"""Evaluate the released skeleton-plus-CLIP interaction ensemble."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from cross_validate_interaction_rgb_fusion import visual_features
from pet_recognition.datasets.interaction import (
    INTERACTION_CLASSES, InteractionSkeletonDataset, summarize_relation_features,
)
from pet_recognition.evaluation.classification_metrics import (
    classification_metrics, save_classification_report,
)


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", default=str(project / "checkpoints/best_models/interaction"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--skeleton-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.model_root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    alpha = float(config["visual_fusion_weight"])
    skeleton_models = [joblib.load(path)["model"] for path in
                       sorted((root / "checkpoints").glob("fold_*.joblib"))]
    rgb_models = [joblib.load(path)["model"] for path in
                  sorted((root / "rgb").glob("fold_*.joblib"))]
    if len(skeleton_models) != 5 or len(rgb_models) != 5:
        raise FileNotFoundError("the release must contain five skeleton and five RGB folds")

    dataset = InteractionSkeletonDataset(
        args.manifest, args.skeleton_dir, args.split, training=False
    )
    relation, targets = [], []
    for item in dataset:
        relation.append(summarize_relation_features(
            item["relation_features"].numpy(), item["frame_mask"].numpy()
        ))
        targets.append(item["label"].item())
    relation = np.stack(relation)
    visual = visual_features(dataset.rows, args.feature_dir, "raw")
    skeleton = np.mean([model.predict_proba(relation) for model in skeleton_models], axis=0)
    rgb = np.mean([model.predict_proba(visual) for model in rgb_models], axis=0)
    probabilities = softmax(np.log(skeleton + 1e-6) + alpha * np.log(rgb + 1e-6))
    predictions = probabilities.argmax(1)
    report = classification_metrics(targets, predictions, INTERACTION_CLASSES)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"ensemble_{args.split}"
    save_classification_report(report, INTERACTION_CLASSES, output, prefix)
    with (output / f"{prefix}_predictions.csv").open(
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
    print(json.dumps({"visual_fusion_weight": alpha, **report}, indent=2))


if __name__ == "__main__":
    main()
