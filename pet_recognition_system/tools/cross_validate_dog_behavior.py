"""Run dog-group cross-validation without touching the frozen test split."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from pet_recognition.datasets.dog_behavior import DOG_BEHAVIOR_CLASSES


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(project / "outputs" / "dog_behavior_manifest_cv.csv"))
    parser.add_argument("--skeleton-dir", required=True)
    parser.add_argument("--model", default="motion_aux_tcn",
                        choices=("motion_aux_tcn",))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--target-frames", type=int, default=32)
    parser.add_argument("--seed-base", type=int, default=28)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    args = parser.parse_args()

    output = (Path(args.output_dir) if args.output_dir else
              project / "work_dirs" / f"dog_behavior_cv_{args.model}")
    output.mkdir(parents=True, exist_ok=True)
    with Path(args.manifest).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        source_rows = list(reader)
        fields = list(reader.fieldnames or ())

    reports = []
    for fold in range(args.folds):
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_manifest = fold_dir / "manifest.csv"
        rows = []
        for source in source_rows:
            row = dict(source)
            if row["split"] == "dev":
                row["split"] = "val" if row["cv_fold"] == str(fold) else "train"
            rows.append(row)
        with fold_manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        command = [
            sys.executable, str(project / "tools" / "train_dog_behavior.py"),
            "--manifest", str(fold_manifest),
            "--skeleton-dir", str(args.skeleton_dir),
            "--model", args.model,
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--output-dir", str(fold_dir),
            "--device", args.device,
            "--seed", str(args.seed_base + fold),
            "--target-frames", str(args.target_frames),
        ]
        print(f"starting fold {fold}", flush=True)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(project)
        subprocess.run(command, check=True, cwd=project, env=environment)
        reports.append(json.loads((fold_dir / "val_metrics.json").read_text(encoding="utf-8")))

    summary = {
        "model": args.model,
        "folds": args.folds,
        "accuracy_mean": float(np.mean([r["accuracy"] for r in reports])),
        "accuracy_std": float(np.std([r["accuracy"] for r in reports])),
        "macro_f1_mean": float(np.mean([r["macro_f1"] for r in reports])),
        "macro_f1_std": float(np.std([r["macro_f1"] for r in reports])),
        "per_class": {},
    }
    for class_name in DOG_BEHAVIOR_CLASSES:
        summary["per_class"][class_name] = {}
        for metric in ("precision", "recall", "f1"):
            values = [r["per_class"][class_name][metric] for r in reports]
            summary["per_class"][class_name][f"{metric}_mean"] = float(np.mean(values))
            summary["per_class"][class_name][f"{metric}_std"] = float(np.std(values))
        summary["per_class"][class_name]["support"] = int(sum(
            r["per_class"][class_name]["support"] for r in reports
        ))
    (output / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
