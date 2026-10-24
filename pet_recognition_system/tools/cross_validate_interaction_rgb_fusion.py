"""Leakage-safe fusion of frozen CLIP features with interaction relations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cross_validate_interaction_relation import connected_groups
from pet_recognition.datasets.interaction import (
    CLASS_TO_INDEX, INTERACTION_CLASSES, InteractionSkeletonDataset,
    summarize_relation_features,
)
from pet_recognition.evaluation.classification_metrics import (
    classification_metrics, save_classification_report,
)


def visual_features(rows, feature_dir, mode):
    features = []
    for row in rows:
        with np.load(Path(feature_dir) / f"{row['clip_id']}.npz", allow_pickle=False) as archive:
            if mode == "prompt":
                values = []
                for name in ("crop_prompt_scores", "full_prompt_scores"):
                    scores = archive[name].astype(np.float32)
                    values.extend((scores.mean(0), scores.max(0), scores.std(0)))
                features.append(np.concatenate(values))
            else:
                values = []
                for name in ("crop_embeddings", "full_embeddings"):
                    embeddings = archive[name].astype(np.float32)
                    values.extend((embeddings.mean(0), embeddings.std(0),
                                   np.abs(embeddings[-1] - embeddings[0])))
                features.append(np.concatenate(values))
    return np.stack(features)


def make_candidates(seed):
    candidates = []
    for c_value in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        candidates.append((f"prompt_logistic_c{c_value}", "prompt", lambda s, c=c_value:
            make_pipeline(StandardScaler(), LogisticRegression(
                C=c, class_weight="balanced", max_iter=5000, random_state=s))))
    for depth in (2, 3, 4, None):
        candidates.append((f"prompt_extra_d{depth}", "prompt", lambda s, d=depth:
            ExtraTreesClassifier(n_estimators=500, max_depth=d, min_samples_leaf=2,
                max_features="sqrt", class_weight="balanced", random_state=s, n_jobs=-1)))
    for components in (4, 8, 16, 24, 32):
        for c_value in (0.003, 0.01, 0.03, 0.1, 0.3):
            candidates.append((f"raw_pca{components}_c{c_value}", "raw", lambda s, n=components, c=c_value:
                make_pipeline(StandardScaler(), PCA(n_components=n, whiten=True, random_state=s),
                    LogisticRegression(C=c, class_weight="balanced", max_iter=5000,
                                       random_state=s))))
    return candidates


def metrics(labels, probabilities):
    return classification_metrics(labels, probabilities.argmax(1), INTERACTION_CLASSES)


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--skeleton-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--base-oof-dir", required=True)
    parser.add_argument("--base-final-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--deployment-alpha", type=float,
                        help="override the validation-selected visual fusion weight")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dev_sets = [InteractionSkeletonDataset(args.manifest, args.skeleton_dir, split,
                                            training=False, seed=args.seed)
                for split in ("train", "val")]
    references = [(dataset, index, row) for dataset in dev_sets
                  for index, row in enumerate(dataset.rows)]
    rows = [reference[2] for reference in references]
    labels = np.asarray([CLASS_TO_INDEX[row["interaction"]] for row in rows])
    groups = connected_groups(rows)
    relation = np.stack([summarize_relation_features(
        dataset[index]["relation_features"].numpy(), dataset[index]["frame_mask"].numpy())
        for dataset, index, _ in references])
    visual = {mode: visual_features(rows, args.feature_dir, mode)
              for mode in ("prompt", "raw")}
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(relation, labels, groups))

    base_oof = np.zeros((len(rows), len(INTERACTION_CLASSES)), dtype=np.float64)
    for fold, (_, val_indices) in enumerate(splits, start=1):
        model = joblib.load(Path(args.base_oof_dir) / f"fold_{fold}.joblib")["model"]
        base_oof[val_indices] = model.predict_proba(relation[val_indices])
    base_report = metrics(labels, base_oof)
    best = {
        "score": (base_report["macro_f1"], base_report["accuracy"]),
        "name": "none", "mode": "prompt", "kind": "weighted", "alpha": 0.0,
        "stack_c": None, "rgb_oof": None, "fused_oof": base_oof,
        "report": base_report,
    }
    for name, mode, factory in make_candidates(args.seed):
        rgb_oof = np.zeros_like(base_oof)
        models = []
        for fold, (train_indices, val_indices) in enumerate(splits, start=1):
            model = factory(args.seed + fold).fit(visual[mode][train_indices], labels[train_indices])
            rgb_oof[val_indices] = model.predict_proba(visual[mode][val_indices])
            models.append(model)
        for alpha in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0):
            fused = np.log(base_oof + 1e-6) + alpha * np.log(rgb_oof + 1e-6)
            report = metrics(labels, fused)
            score = (report["macro_f1"], report["accuracy"])
            if score > best["score"]:
                best = {"score": score, "name": name, "mode": mode,
                        "kind": "weighted", "alpha": alpha, "stack_c": None,
                        "rgb_oof": rgb_oof, "fused_oof": fused, "report": report}
        stacked_x = np.concatenate((np.log(base_oof + 1e-6),
                                    np.log(rgb_oof + 1e-6)), axis=1)
        for stack_c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            stacked_oof = np.zeros_like(base_oof)
            for fold, (train_indices, val_indices) in enumerate(splits, start=1):
                stacker = make_pipeline(StandardScaler(), LogisticRegression(
                    C=stack_c, class_weight="balanced", max_iter=5000,
                    random_state=args.seed + fold))
                stacker.fit(stacked_x[train_indices], labels[train_indices])
                stacked_oof[val_indices] = stacker.predict_proba(stacked_x[val_indices])
            report = metrics(labels, stacked_oof)
            score = (report["macro_f1"], report["accuracy"])
            if score > best["score"]:
                best = {"score": score, "name": name, "mode": mode,
                        "kind": "stacked", "alpha": None, "stack_c": stack_c,
                        "rgb_oof": rgb_oof, "fused_oof": stacked_oof,
                        "report": report}
        print(f"checked {name}", flush=True)

    name, mode = best["name"], best["mode"]
    alpha, stack_c = best["alpha"], best["stack_c"]
    fused_oof, oof_report = best["fused_oof"], best["report"]
    rgb_oof_report = (metrics(labels, best["rgb_oof"])
                      if best["rgb_oof"] is not None else base_report)
    oof_sensitivity = []
    if best["rgb_oof"] is not None:
        for diagnostic_alpha in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
            diagnostic = (np.log(base_oof + 1e-6) +
                          diagnostic_alpha * np.log(best["rgb_oof"] + 1e-6))
            diagnostic_report = metrics(labels, diagnostic)
            oof_sensitivity.append({
                "alpha": diagnostic_alpha,
                "accuracy": diagnostic_report["accuracy"],
                "macro_f1": diagnostic_report["macro_f1"],
            })
    save_classification_report(base_report, INTERACTION_CLASSES, output, "base_oof")
    save_classification_report(oof_report, INTERACTION_CLASSES, output, "fused_oof")

    fixed_val_sensitivity = []
    deployment_alpha = alpha
    final_templates = [joblib.load(path)["model"] for path in
                       sorted(Path(args.base_final_dir).glob("fold_*.joblib"))]
    if name != "none" and best["kind"] == "weighted":
        train_count = len(dev_sets[0])
        train_indices = np.arange(train_count)
        val_indices = np.arange(train_count, len(rows))
        fixed_base_models = [clone(model).fit(relation[train_indices], labels[train_indices])
                             for model in final_templates]
        fixed_base_val = np.mean(
            [model.predict_proba(relation[val_indices]) for model in fixed_base_models], axis=0)
        selected = next(item for item in make_candidates(args.seed) if item[0] == name)
        factory = selected[2]
        fixed_rgb_models = [factory(args.seed + offset).fit(
            visual[mode][train_indices], labels[train_indices]) for offset in range(1, 6)]
        fixed_rgb_val = np.mean(
            [model.predict_proba(visual[mode][val_indices]) for model in fixed_rgb_models], axis=0)
        best_validation = None
        for candidate_alpha in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0):
            probability = (np.log(fixed_base_val + 1e-6) +
                           candidate_alpha * np.log(fixed_rgb_val + 1e-6))
            report = metrics(labels[val_indices], probability)
            fixed_val_sensitivity.append({"alpha": candidate_alpha,
                                          "accuracy": report["accuracy"],
                                          "macro_f1": report["macro_f1"]})
            key = (report["macro_f1"], report["accuracy"], -candidate_alpha)
            if best_validation is None or key > best_validation[0]:
                best_validation = (key, candidate_alpha)
        deployment_alpha = best_validation[1]
    if args.deployment_alpha is not None:
        deployment_alpha = args.deployment_alpha
    if name != "none" and best["kind"] == "weighted":
        deployment_oof = (np.log(base_oof + 1e-6) +
                          deployment_alpha * np.log(best["rgb_oof"] + 1e-6))
        deployment_oof_report = metrics(labels, deployment_oof)
    else:
        deployment_oof_report = oof_report
    save_classification_report(
        deployment_oof_report, INTERACTION_CLASSES, output, "fused_oof"
    )

    test = InteractionSkeletonDataset(args.manifest, args.skeleton_dir, "test", training=False)
    test_rows = test.rows
    test_labels = np.asarray([item["label"].item() for item in test])
    test_relation = np.stack([summarize_relation_features(
        item["relation_features"].numpy(), item["frame_mask"].numpy()) for item in test])
    final_models = final_templates
    base_test = np.mean([model.predict_proba(test_relation) for model in final_models], axis=0)
    base_test_report = metrics(test_labels, base_test)
    if name == "none":
        fused_test, final_rgb_models, final_stacker = base_test, [], None
        sensitivity = []
        rgb_test_report = base_test_report
    else:
        selected = next(item for item in make_candidates(args.seed) if item[0] == name)
        factory = selected[2]
        final_rgb_models = [factory(args.seed + offset).fit(visual[mode], labels)
                            for offset in range(1, 6)]
        test_visual = visual_features(test_rows, args.feature_dir, mode)
        rgb_test = np.mean([model.predict_proba(test_visual) for model in final_rgb_models], axis=0)
        rgb_test_report = metrics(test_labels, rgb_test)
        sensitivity = []
        for diagnostic_alpha in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
            diagnostic_probability = (np.log(base_test + 1e-6) +
                                      diagnostic_alpha * np.log(rgb_test + 1e-6))
            diagnostic_report = metrics(test_labels, diagnostic_probability)
            sensitivity.append({
                "alpha": diagnostic_alpha,
                "accuracy": diagnostic_report["accuracy"],
                "macro_f1": diagnostic_report["macro_f1"],
                "per_class_recall": {
                    label: diagnostic_report["per_class"][label]["recall"]
                    for label in INTERACTION_CLASSES
                },
            })
        if best["kind"] == "stacked":
            stacked_dev = np.concatenate((np.log(base_oof + 1e-6),
                                          np.log(best["rgb_oof"] + 1e-6)), axis=1)
            final_stacker = make_pipeline(StandardScaler(), LogisticRegression(
                C=stack_c, class_weight="balanced", max_iter=5000,
                random_state=args.seed))
            final_stacker.fit(stacked_dev, labels)
            stacked_test = np.concatenate((np.log(base_test + 1e-6),
                                           np.log(rgb_test + 1e-6)), axis=1)
            fused_test = final_stacker.predict_proba(stacked_test)
        else:
            final_stacker = None
            fused_test = (np.log(base_test + 1e-6) +
                          deployment_alpha * np.log(rgb_test + 1e-6))
    fused_test_report = metrics(test_labels, fused_test)
    save_classification_report(base_test_report, INTERACTION_CLASSES, output, "base_test")
    save_classification_report(fused_test_report, INTERACTION_CLASSES, output, "fused_test")
    for fold, model in enumerate(final_rgb_models, start=1):
        joblib.dump({"model": model, "mode": mode, "alpha": deployment_alpha,
                     "name": name},
                    output / f"rgb_fold_{fold}.joblib")
    if final_stacker is not None:
        joblib.dump({"model": final_stacker, "name": name, "mode": mode,
                     "stack_c": stack_c}, output / "stacker.joblib")
    summary = {
        "selection_protocol": "grouped OOF only; test evaluated after selection",
        "model": name, "mode": mode, "fusion": best["kind"],
        "oof_selected_alpha": alpha, "validation_selected_alpha": deployment_alpha,
        "stack_c": stack_c,
        "base_oof": {"accuracy": base_report["accuracy"], "macro_f1": base_report["macro_f1"]},
        "rgb_only_oof": {"accuracy": rgb_oof_report["accuracy"],
                         "macro_f1": rgb_oof_report["macro_f1"]},
        "oof_search_best": {"accuracy": oof_report["accuracy"],
                            "macro_f1": oof_report["macro_f1"]},
        "fused_oof": {"accuracy": deployment_oof_report["accuracy"],
                      "macro_f1": deployment_oof_report["macro_f1"]},
        "oof_alpha_sensitivity": oof_sensitivity,
        "fixed_validation_alpha_sensitivity": fixed_val_sensitivity,
        "base_test": {"accuracy": base_test_report["accuracy"], "macro_f1": base_test_report["macro_f1"]},
        "rgb_only_test": {"accuracy": rgb_test_report["accuracy"],
                          "macro_f1": rgb_test_report["macro_f1"]},
        "fused_test": {"accuracy": fused_test_report["accuracy"], "macro_f1": fused_test_report["macro_f1"]},
        "test_alpha_sensitivity_diagnostic": sensitivity,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
