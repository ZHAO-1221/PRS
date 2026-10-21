"""Fuse five-class TCN folds with geometry and pairwise specialists."""
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

from cross_validate_dog_joint_specialist import combined_features
from pet_recognition.datasets.dog_behavior import DOG_BEHAVIOR_CLASSES, DogBehaviorSkeletonDataset
from pet_recognition.evaluation.classification_metrics import classification_metrics, save_classification_report
from train_dog_play_specialist import tcn_probabilities
from train_dog_eating_specialist import dataset_features as eating_features


INDEX = {name: index for index, name in enumerate(DOG_BEHAVIOR_CLASSES)}
PAIRS = (
    ("standing", "walking_run"),
    ("standing", "sitting"),
    ("standing", "lying_sleeping"),
    ("standing", "eating_drinking"),
    ("eating_drinking", "lying_sleeping"),
    ("eating_drinking", "sitting"),
)


def report(folds, logits):
    labels = np.concatenate([fold["val_labels"] for fold in folds])
    prediction = np.concatenate([value.argmax(1) for value in logits])
    return classification_metrics(labels, prediction, DOG_BEHAVIOR_CLASSES)


def multiclass_factories(seed):
    for depth in (2, 3, 4, 5, None):
        for leaf in (1, 2, 3):
            yield f"extra_d{depth}_l{leaf}", lambda offset, d=depth, l=leaf: ExtraTreesClassifier(
                n_estimators=600, max_depth=d, min_samples_leaf=l, max_features="sqrt",
                class_weight="balanced", n_jobs=-1, random_state=seed + offset)
    for leaf in (1, 2, 3):
        yield f"forest_l{leaf}", lambda offset, l=leaf: RandomForestClassifier(
            n_estimators=600, min_samples_leaf=l, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed + offset)
    for c_value in (0.01, 0.03, 0.1, 0.3, 1.0):
        yield f"logistic_c{c_value}", lambda offset, c=c_value: make_pipeline(
            StandardScaler(), LogisticRegression(C=c, class_weight="balanced",
                                                   max_iter=4000, random_state=seed + offset))


def pair_model(seed, depth, leaf):
    return ExtraTreesClassifier(
        n_estimators=250, max_depth=depth, min_samples_leaf=leaf,
        max_features=0.75, class_weight="balanced", n_jobs=-1, random_state=seed)


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-dir", required=True)
    parser.add_argument("--skeleton-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--feature-version", default="fixed_region_v2",
                        choices=("head_duration_v1", "fixed_region_v2"))
    args = parser.parse_args()

    folds = []
    for fold_index in range(args.folds):
        fold_dir = Path(args.cv_dir) / f"fold_{fold_index}"
        common = dict(manifest=fold_dir / "manifest.csv", skeleton_dir=args.skeleton_dir,
                      target_frames=32, training=False, augment=False)
        train = DogBehaviorSkeletonDataset(split="train", **common)
        val = DogBehaviorSkeletonDataset(split="val", **common)
        train_x, train_labels = combined_features(train, args.feature_version)
        val_x, val_labels = combined_features(val, args.feature_version)
        include_fixed_region = args.feature_version == "fixed_region_v2"
        train_eating, eating_train_labels = eating_features(
            train, include_fixed_region=include_fixed_region)
        val_eating, eating_val_labels = eating_features(
            val, include_fixed_region=include_fixed_region)
        if not (np.array_equal(train_labels, eating_train_labels) and
                np.array_equal(val_labels, eating_val_labels)):
            raise AssertionError("eating feature order mismatch")
        probability = tcn_probabilities(fold_dir / "best.pt", val, args.device)
        folds.append({"train_x": train_x, "train_labels": train_labels,
                      "val_x": val_x, "val_labels": val_labels,
                      "train_eating": train_eating, "val_eating": val_eating,
                      "logits": np.log(probability + 1e-8)})
        print(f"loaded fold {fold_index}", flush=True)

    base_logits = [fold["logits"] for fold in folds]
    base = report(folds, base_logits)
    best = (base["macro_f1"], base["accuracy"], "none", 0.0, None, base, base_logits)
    for name, factory in multiclass_factories(args.seed):
        models, probabilities = [], []
        for fold_index, fold in enumerate(folds):
            model = factory(fold_index).fit(fold["train_x"], fold["train_labels"])
            models.append(model)
            probabilities.append(model.predict_proba(fold["val_x"]))
        for alpha in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
            logits = [base_value + alpha * np.log(probability + 1e-5)
                      for base_value, probability in zip(base_logits, probabilities)]
            current = report(folds, logits)
            key = (current["macro_f1"], current["accuracy"])
            if key > best[:2]:
                best = (*key, name, alpha, models, current, logits)
        print(f"checked {name}", flush=True)

    _, _, model_name, model_alpha, models, current_report, current_logits = best
    stages = []
    # Add only pairwise stages that strictly improve aggregate OOF macro-F1.
    for stage_index in range(2):
        stage_best = (current_report["macro_f1"], current_report["accuracy"], None)
        for positive, negative in PAIRS:
            pi, ni = INDEX[positive], INDEX[negative]
            for depth in (2, 3, None):
                for leaf in (1, 2):
                    pair_models, decisions = [], []
                    for fold_index, fold in enumerate(folds):
                        keep = np.isin(fold["train_labels"], (pi, ni))
                        model = pair_model(args.seed + 100 * stage_index + fold_index,
                                           depth, leaf).fit(
                            fold["train_x"][keep], fold["train_labels"][keep] == pi)
                        probability = model.predict_proba(fold["val_x"])[:, 1]
                        decisions.append(np.log((probability + 0.02) /
                                                (1 - probability + 0.02)))
                        pair_models.append(model)
                    for gate in ("all", "pair_top", "close"):
                        margins = (0.5, 1.0, 1.5, 2.0) if gate == "close" else (0.0,)
                        for margin in margins:
                            for alpha in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
                                output = []
                                for logits, decision in zip(current_logits, decisions):
                                    value = logits.copy()
                                    if gate == "pair_top":
                                        ranking = np.argsort(value, axis=1)[:, -2:]
                                        enabled = np.logical_and((ranking == pi).any(1),
                                                                 (ranking == ni).any(1))
                                    elif gate == "close":
                                        enabled = np.abs(value[:, pi] - value[:, ni]) <= margin
                                    else:
                                        enabled = np.ones(len(value), dtype=bool)
                                    value[enabled, pi] += alpha * decision[enabled]
                                    value[enabled, ni] -= alpha * decision[enabled]
                                    output.append(value)
                                candidate = report(folds, output)
                                key = (candidate["macro_f1"], candidate["accuracy"])
                                if key > stage_best[:2]:
                                    stage_best = (*key, {"positive": positive,
                                        "negative": negative, "depth": depth, "leaf": leaf,
                                        "gate": gate, "margin": margin, "alpha": alpha,
                                        "models": pair_models, "logits": output,
                                        "report": candidate})
        selected = stage_best[2]
        if selected is None:
            break
        stages.append(selected)
        current_logits, current_report = selected["logits"], selected["report"]
        print("selected", {k: v for k, v in selected.items()
                           if k not in ("models", "logits", "report")}, flush=True)

    # A dedicated head-duration specialist distinguishes sustained feeding
    # posture from a brief glance/sniff.  Keep it only when OOF macro-F1 rises.
    eating_index = INDEX["eating_drinking"]
    duration_best = (current_report["macro_f1"], current_report["accuracy"], None)
    for depth in (2, 3, None):
        for leaf in (1, 2, 3):
            duration_models, decisions = [], []
            for fold_index, fold in enumerate(folds):
                model = ExtraTreesClassifier(
                    n_estimators=500, max_depth=depth, min_samples_leaf=leaf,
                    max_features="sqrt", class_weight="balanced", n_jobs=-1,
                    random_state=args.seed + 500 + fold_index,
                ).fit(fold["train_eating"],
                      fold["train_labels"] == eating_index)
                probability = model.predict_proba(fold["val_eating"])[:, 1]
                decisions.append(np.log((probability + 0.02) /
                                        (1 - probability + 0.02)))
                duration_models.append(model)
            for gate in ("all", "close"):
                margins = (0.5, 1.0, 1.5, 2.0) if gate == "close" else (0.0,)
                for margin in margins:
                    for alpha in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
                        output = []
                        for logits, decision in zip(current_logits, decisions):
                            value = logits.copy()
                            if gate == "close":
                                other = np.max(np.delete(value, eating_index, axis=1), axis=1)
                                enabled = np.abs(value[:, eating_index] - other) <= margin
                            else:
                                enabled = np.ones(len(value), dtype=bool)
                            value[enabled, eating_index] += alpha * decision[enabled]
                            output.append(value)
                        candidate = report(folds, output)
                        key = (candidate["macro_f1"], candidate["accuracy"])
                        if key > duration_best[:2]:
                            duration_best = (*key, {
                                "depth": depth, "leaf": leaf, "gate": gate,
                                "margin": margin, "alpha": alpha,
                                "models": duration_models, "logits": output,
                                "report": candidate,
                            })
    duration_stage = duration_best[2]
    if duration_stage is not None:
        current_logits = duration_stage["logits"]
        current_report = duration_stage["report"]
        print("selected eating duration", {k: v for k, v in duration_stage.items()
              if k not in ("models", "logits", "report")}, flush=True)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if models is not None:
        for fold_index, model in enumerate(models):
            joblib.dump({"model": model, "alpha": model_alpha, "name": model_name,
                         "classes": DOG_BEHAVIOR_CLASSES},
                        output / f"fold_{fold_index}_multiclass.joblib")
    for stage_index, stage in enumerate(stages, 1):
        metadata = {k: v for k, v in stage.items()
                    if k not in ("models", "logits", "report")}
        for fold_index, model in enumerate(stage["models"]):
            joblib.dump({"model": model, **metadata},
                        output / f"stage_{stage_index}_fold_{fold_index}.joblib")
    if duration_stage is not None:
        metadata = {k: v for k, v in duration_stage.items()
                    if k not in ("models", "logits", "report")}
        for fold_index, model in enumerate(duration_stage["models"]):
            joblib.dump({"model": model, **metadata},
                        output / f"fold_{fold_index}_eating_duration.joblib")
    save_classification_report(current_report, DOG_BEHAVIOR_CLASSES, output, "cv_fused")
    summary = {"base_accuracy": base["accuracy"], "base_macro_f1": base["macro_f1"],
               "final_accuracy": current_report["accuracy"],
               "final_macro_f1": current_report["macro_f1"],
               "multiclass_model": model_name, "multiclass_alpha": model_alpha,
               "feature_version": args.feature_version,
               "pairwise_stages": [{k: v for k, v in stage.items()
                                     if k not in ("models", "logits", "report")}
                                    for stage in stages],
               "eating_duration_stage": ({k: v for k, v in duration_stage.items()
                   if k not in ("models", "logits", "report")}
                   if duration_stage is not None else None)}
    (output / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
