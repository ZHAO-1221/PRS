"""Leakage-safe grouped cross-validation and fold ensembling for interaction recognition."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold

from pet_recognition.datasets.interaction import (
    CLASS_TO_INDEX, INTERACTION_CLASSES, InteractionSkeletonDataset,
    summarize_relation_features,
)
from pet_recognition.evaluation.classification_metrics import (
    classification_metrics, save_classification_report,
)


def connected_groups(rows):
    """Return one component id per row for the human/dog bipartite graph."""
    parent = {}

    def find(item):
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for row in rows:
        union("d:" + row["dog_id"], "p:" + row["person_id"])
    roots = {find("d:" + row["dog_id"]) for row in rows}
    root_to_index = {root: index for index, root in enumerate(sorted(roots))}
    return np.asarray([root_to_index[find("d:" + row["dog_id"])] for row in rows])


def parse_max_features(value):
    if value.lower() in {"none", "all"}:
        return None
    if value in {"sqrt", "log2"}:
        return value
    return float(value)


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("PET_DATA_ROOT"))
    parser.add_argument("--skeleton-dir")
    parser.add_argument("--manifest", default=str(project / "outputs" / "interaction_manifest.csv"))
    parser.add_argument("--output-dir", default=str(project / "work_dirs" / "interaction_relation_cv5"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--augmentations", type=int, default=4)
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
    if not args.data_root and not args.skeleton_dir:
        parser.error("set PET_DATA_ROOT/--data-root or pass --skeleton-dir")
    skeleton_dir = (Path(args.skeleton_dir) if args.skeleton_dir
                    else Path(args.data_root) / "skeletons_generated")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    max_features = parse_max_features(args.max_features)
    class_weight = None if args.class_weight == "none" else args.class_weight
    if args.hand_feeding_weight != 1.0:
        class_weight = {0: 1.0, 1: args.hand_feeding_weight, 2: 1.0, 3: 1.0}

    datasets = [
        InteractionSkeletonDataset(args.manifest, skeleton_dir, split, training=False,
                                   seed=args.seed)
        for split in ("train", "val")
    ]
    references = [(dataset, index, row) for dataset in datasets
                  for index, row in enumerate(dataset.rows)]
    rows = [reference[2] for reference in references]
    labels = np.asarray([CLASS_TO_INDEX[row["interaction"]] for row in rows])
    groups = connected_groups(rows)
    if len(np.unique(groups)) < args.folds:
        raise ValueError(f"only {len(np.unique(groups))} connected identity groups")

    base_features = []
    for dataset, index, _ in references:
        item = dataset[index]
        base_features.append(summarize_relation_features(
            item["relation_features"].numpy(), item["frame_mask"].numpy()
        ))
    base_features = np.stack(base_features)

    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    predictions = np.full(len(rows), -1, dtype=np.int64)
    probabilities = np.zeros((len(rows), len(INTERACTION_CLASSES)), dtype=np.float64)
    fold_reports, fold_assignments = [], np.full(len(rows), -1, dtype=np.int64)
    for fold, (train_indices, val_indices) in enumerate(
        splitter.split(base_features, labels, groups), start=1
    ):
        train_x = [base_features[index] for index in train_indices]
        train_y = [labels[index] for index in train_indices]
        for augmentation in range(1, args.augmentations + 1):
            for dataset in datasets:
                dataset.training = True; dataset.strong_augmentation = True
                dataset.set_epoch(fold * 100 + augmentation)
            for index in train_indices:
                dataset, dataset_index, _ = references[index]
                item = dataset[dataset_index]
                train_x.append(summarize_relation_features(
                    item["relation_features"].numpy(), item["frame_mask"].numpy()
                ))
                train_y.append(labels[index])
        for dataset in datasets:
            dataset.training = False; dataset.strong_augmentation = False
        model = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            max_depth=None,
            min_samples_leaf=args.min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=args.seed + fold,
        )
        model.fit(np.stack(train_x), np.asarray(train_y))
        fold_probability = model.predict_proba(base_features[val_indices])
        fold_prediction = model.classes_[fold_probability.argmax(axis=1)]
        probabilities[val_indices] = fold_probability
        predictions[val_indices] = fold_prediction
        fold_assignments[val_indices] = fold
        report = classification_metrics(labels[val_indices], fold_prediction,
                                        INTERACTION_CLASSES)
        fold_reports.append(report)
        joblib.dump({"model": model, "classes": INTERACTION_CLASSES,
                     "summary_dim": base_features.shape[1], "fold": fold},
                    output / f"fold_{fold}.joblib")
        save_classification_report(report, INTERACTION_CLASSES, output, f"fold_{fold}_val")

    overall = classification_metrics(labels, predictions, INTERACTION_CLASSES)
    save_classification_report(overall, INTERACTION_CLASSES, output, "oof")
    accuracy = np.asarray([report["accuracy"] for report in fold_reports])
    macro_f1 = np.asarray([report["macro_f1"] for report in fold_reports])
    summary = {
        "protocol": "5-fold stratified connected-human/dog-group CV on original train+val only",
        "clips": len(rows), "identity_components": int(len(np.unique(groups))),
        "augmentations_per_training_clip": args.augmentations,
        "model_parameters": {
            "n_estimators": args.n_estimators,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "class_weight": args.class_weight,
            "hand_feeding_weight": args.hand_feeding_weight,
        },
        "accuracy_mean": float(accuracy.mean()), "accuracy_std": float(accuracy.std()),
        "macro_f1_mean": float(macro_f1.mean()), "macro_f1_std": float(macro_f1.std()),
        "out_of_fold": overall,
    }
    (output / "cv_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("clip_id", "dog_id", "person_id", "fold", "actual", "predicted"))
        for row, fold, target, prediction in zip(rows, fold_assignments, labels, predictions):
            writer.writerow((row["clip_id"], row["dog_id"], row["person_id"], fold,
                             INTERACTION_CLASSES[target], INTERACTION_CLASSES[prediction]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
