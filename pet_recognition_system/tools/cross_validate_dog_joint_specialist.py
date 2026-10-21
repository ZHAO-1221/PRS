"""Fuse a joint engineered-feature classifier with the frozen five-fold model.

The earlier specialists answer one binary question at a time.  This stage learns
the competing six-way interpretation of the same geometry, which is especially
useful for eating-vs-sitting/lying and playing-vs-walking errors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pet_recognition.datasets.dog_behavior import DOG_BEHAVIOR_CLASSES, DogBehaviorSkeletonDataset
from pet_recognition.evaluation.classification_metrics import classification_metrics, save_classification_report
from train_dog_eating_specialist import dataset_features as eating_features
from train_dog_play_specialist import dataset_features as play_features
from train_dog_play_specialist import tcn_probabilities
from train_dog_standing_specialist import dataset_features as standing_features
from train_dog_walking_specialist import dataset_features as walking_features


INDEX = {name: DOG_BEHAVIOR_CLASSES.index(name) for name in DOG_BEHAVIOR_CLASSES}


def combined_features(dataset, feature_version="fixed_region_v2"):
    if feature_version not in ("head_duration_v1", "fixed_region_v2"):
        raise ValueError(f"unknown feature version: {feature_version}")
    blocks = []
    labels = None
    functions = (
        play_features,
        standing_features,
        lambda value: eating_features(
            value, include_fixed_region=feature_version == "fixed_region_v2"),
        walking_features,
    )
    for function in functions:
        values, current = function(dataset)
        if labels is not None and not np.array_equal(labels, current):
            raise AssertionError("feature order mismatch")
        labels = current
        blocks.append(values)
    return np.concatenate(blocks, axis=1), labels


def add_binary(logits, features, artifact, class_name):
    probability = artifact["model"].predict_proba(features)[:, 1]
    logits[:, INDEX[class_name]] += artifact["alpha"] * np.log(
        (probability + 0.02) / (1 - probability + 0.02)
    )
    return logits


def aggregate(folds, logits):
    labels = np.concatenate([fold["val_labels"] for fold in folds])
    prediction = np.concatenate([value.argmax(1) for value in logits])
    return classification_metrics(labels, prediction, DOG_BEHAVIOR_CLASSES)


def candidates(seed):
    for depth in (3, 5, None):
        for leaf in (1, 2, 3):
            yield (f"extra_d{depth}_l{leaf}", lambda offset, d=depth, l=leaf:
                   ExtraTreesClassifier(n_estimators=500, max_depth=d,
                                        min_samples_leaf=l, max_features="sqrt",
                                        class_weight="balanced", n_jobs=-1,
                                        random_state=seed + offset))
    for leaf in (1, 2, 3):
        yield (f"forest_l{leaf}", lambda offset, l=leaf:
               RandomForestClassifier(n_estimators=500, min_samples_leaf=l,
                                      max_features="sqrt", class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=seed + offset))
    for c_value in (0.03, 0.1, 0.3, 1.0):
        yield (f"logistic_c{c_value}", lambda offset, c=c_value:
               make_pipeline(StandardScaler(), LogisticRegression(
                   C=c, class_weight="balanced", max_iter=3000,
                   random_state=seed + offset)))


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-dir", default=str(project / "work_dirs" / "dog_behavior_cv_motion_aux_tcn_v1"))
    parser.add_argument("--specialist-dir", default=str(project / "work_dirs" / "dog_behavior_cv_motion_aux_specialists_v1"))
    parser.add_argument("--eating-dir", default=str(project / "work_dirs" / "dog_behavior_cv_motion_aux_eating_v1"))
    parser.add_argument("--walking-dir", default=str(project / "work_dirs" / "dog_behavior_cv_motion_aux_walking_v1"))
    parser.add_argument("--skeleton-dir", required=True)
    parser.add_argument("--output-dir", default=str(project / "work_dirs" / "dog_behavior_cv_joint_specialist_v1"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=28)
    args = parser.parse_args()

    cv_dir = Path(args.cv_dir)
    specialist_dir = Path(args.specialist_dir)
    eating_dir = Path(args.eating_dir)
    walking_dir = Path(args.walking_dir)
    folds = []
    for fold_index in range(args.folds):
        fold_dir = cv_dir / f"fold_{fold_index}"
        common = dict(manifest=fold_dir / "manifest.csv", skeleton_dir=args.skeleton_dir,
                      target_frames=32, training=False, augment=False)
        train = DogBehaviorSkeletonDataset(split="train", **common)
        val = DogBehaviorSkeletonDataset(split="val", **common)
        train_x, train_labels = combined_features(train)
        val_x, val_labels = combined_features(val)

        val_play, _ = play_features(val)
        val_standing, _ = standing_features(val)
        val_eating, _ = eating_features(val)
        val_walking, _ = walking_features(val)
        logits = np.log(tcn_probabilities(fold_dir / "best.pt", val, args.device) + 1e-8)
        prior = joblib.load(specialist_dir / f"fold_{fold_index}_specialists.joblib")
        logits = add_binary(logits, val_play, {"model": prior["play_model"],
                            "alpha": prior["play_alpha"]}, "playing_alone")
        logits = add_binary(logits, val_standing, {"model": prior["standing_model"],
                            "alpha": prior["standing_alpha"]}, "standing")
        logits = add_binary(logits, val_eating,
                            joblib.load(eating_dir / f"fold_{fold_index}_eating.joblib"),
                            "eating_drinking")
        logits = add_binary(logits, val_walking,
                            joblib.load(walking_dir / f"fold_{fold_index}_walking.joblib"),
                            "walking_run")
        folds.append(dict(train_x=train_x, train_labels=train_labels,
                          val_x=val_x, val_labels=val_labels, base_logits=logits))
        print(f"loaded fold {fold_index}: train={len(train_labels)} val={len(val_labels)}", flush=True)

    base_logits = [fold["base_logits"] for fold in folds]
    base_report = aggregate(folds, base_logits)
    best = None
    for name, factory in candidates(args.seed):
        models, probabilities = [], []
        for fold_index, fold in enumerate(folds):
            model = factory(fold_index).fit(fold["train_x"], fold["train_labels"])
            models.append(model)
            probabilities.append(model.predict_proba(fold["val_x"]))
        for alpha in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
            fused = [base + alpha * np.log(probability + 1e-5)
                     for base, probability in zip(base_logits, probabilities)]
            report = aggregate(folds, fused)
            # A candidate is accepted only if it improves macro-F1.  Accuracy
            # then breaks ties, preventing gains that come only from a rare class.
            key = (report["macro_f1"] >= base_report["macro_f1"],
                   report["macro_f1"], report["accuracy"])
            if best is None or key > best[0]:
                best = (key, name, alpha, models, report)
        print(f"checked {name}", flush=True)

    _, name, alpha, models, final = best
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for fold_index, model in enumerate(models):
        joblib.dump({"model": model, "alpha": alpha, "name": name,
                     "classes": DOG_BEHAVIOR_CLASSES},
                    output / f"fold_{fold_index}_joint.joblib")
    save_classification_report(final, DOG_BEHAVIOR_CLASSES, output, "cv_fused")
    summary = {"base_accuracy": base_report["accuracy"],
               "base_macro_f1": base_report["macro_f1"],
               "final_accuracy": final["accuracy"],
               "final_macro_f1": final["macro_f1"],
               "model": name, "alpha": alpha}
    (output / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
