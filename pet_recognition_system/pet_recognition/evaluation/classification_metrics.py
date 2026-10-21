"""Deterministic multiclass metrics used by behaviour recognition."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def confusion_matrix(targets, predictions, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def classification_metrics(targets, predictions, class_names):
    matrix = confusion_matrix(targets, predictions, len(class_names))
    support = matrix.sum(axis=1)
    true_positive = np.diag(matrix).astype(np.float64)
    precision = np.divide(true_positive, matrix.sum(axis=0), out=np.zeros_like(true_positive), where=matrix.sum(axis=0) > 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(true_positive), where=(precision + recall) > 0)
    total = int(matrix.sum())
    result = {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {
                "precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
    }
    return result


def save_classification_report(report, class_names, output_dir, prefix="val"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{prefix}_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output / f"{prefix}_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *class_names])
        for name, row in zip(class_names, report["confusion_matrix"]):
            writer.writerow([name, *row])
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    matrix = np.asarray(report["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
             xticklabels=class_names, yticklabels=class_names,
             xlabel="Predicted", ylabel="Actual", title=f"{prefix.title()} confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output / f"{prefix}_confusion_matrix.png", dpi=180)
    plt.close(figure)
