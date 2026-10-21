"""Train the final interaction ensemble on all train and validation clips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from pet_recognition.datasets.interaction import (
    INTERACTION_CLASSES,
    InteractionSkeletonDataset,
    summarize_relation_features,
)


def parse_max_features(value):
    if value.lower() in {"none", "all"}:
        return None
    if value in {"sqrt", "log2"}:
        return value
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--skeleton-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", type=int, default=5)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument(
        "--class-weight",
        choices=("balanced", "balanced_subsample", "none"),
        default="balanced",
    )
    parser.add_argument("--hand-feeding-weight", type=float, default=1.0)
    args = parser.parse_args()

    datasets = [
        InteractionSkeletonDataset(
            args.manifest, args.skeleton_dir, split, training=False, seed=args.seed
        )
        for split in ("train", "val")
    ]
    features, labels = [], []
    for dataset in datasets:
        for item in dataset:
            features.append(
                summarize_relation_features(
                    item["relation_features"].numpy(), item["frame_mask"].numpy()
                )
            )
            labels.append(item["label"].item())
    x, y = np.stack(features), np.asarray(labels)
    max_features = parse_max_features(args.max_features)
    class_weight = None if args.class_weight == "none" else args.class_weight
    if args.hand_feeding_weight != 1.0:
        class_weight = {0: 1.0, 1: args.hand_feeding_weight, 2: 1.0, 3: 1.0}

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for index in range(1, args.models + 1):
        model = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            max_depth=None,
            min_samples_leaf=args.min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=args.seed + index,
        )
        model.fit(x, y)
        joblib.dump(
            {
                "model": model,
                "classes": INTERACTION_CLASSES,
                "summary_dim": x.shape[1],
                "training_protocol": "all_train_val_seed_ensemble",
                "seed": args.seed + index,
            },
            output / f"fold_{index}.joblib",
        )

    summary = {
        "protocol": "full train+val ExtraTrees seed ensemble",
        "clips": len(y),
        "models": args.models,
        "seeds": list(range(args.seed + 1, args.seed + args.models + 1)),
        "model_parameters": {
            "n_estimators": args.n_estimators,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "class_weight": args.class_weight,
            "hand_feeding_weight": args.hand_feeding_weight,
        },
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
