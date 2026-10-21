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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--skeleton-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", type=int, default=5)
    parser.add_argument("--seed", type=int, default=28)
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

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for index in range(1, args.models + 1):
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
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
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
