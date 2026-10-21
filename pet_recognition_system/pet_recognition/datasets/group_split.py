"""Leakage-safe, class-balanced group splitting for behaviour clips."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

SPLITS = ("train", "val", "test")


def balanced_group_assignment(rows, label_key="behavior_label", seed=28, trials=30_000):
    """Assign dog IDs to 70/15/15 splits using a deterministic random search."""
    valid = [row for row in rows if row.get("valid") == "1"]
    dogs = sorted({row["dog_id"] for row in valid})
    labels = sorted({row[label_key] for row in valid})
    if len(dogs) < 3:
        raise ValueError("at least three dog IDs are required")
    dog_counts = {dog: Counter() for dog in dogs}
    for row in valid:
        dog_counts[row["dog_id"]][row[label_key]] += 1
    total = Counter(row[label_key] for row in valid)
    n_val = max(1, round(len(dogs) * 0.15))
    n_test = max(1, round(len(dogs) * 0.15))
    n_train = len(dogs) - n_val - n_test
    group_sizes = (n_train, n_val, n_test)
    ratios = np.asarray(group_sizes, dtype=np.float64) / len(dogs)
    targets = np.asarray([[total[label] * ratio for label in labels] for ratio in ratios])
    target_samples = np.asarray([len(valid) * ratio for ratio in ratios])
    rng = np.random.default_rng(seed)
    best_score, best = float("inf"), None
    for _ in range(trials):
        shuffled = rng.permutation(dogs)
        groups = (
            shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]
        )
        counts = np.asarray([
            [sum(dog_counts[dog][label] for dog in group) for label in labels]
            for group in groups
        ], dtype=np.float64)
        samples = counts.sum(axis=1)
        score = float((((counts - targets) / np.maximum(targets, 1)) ** 2).sum())
        score += float(((((samples - target_samples) / np.maximum(target_samples, 1)) ** 2).sum()))
        # Validation and test must each exercise every class.
        score += 1_000 * float((counts[1:] == 0).sum())
        if score < best_score:
            best_score, best = score, groups
    if best is None:
        raise RuntimeError("failed to find a group split")
    return {dog: split for split, group in zip(SPLITS, best) for dog in group}


def apply_group_assignment(rows, assignment):
    output = [dict(row) for row in rows]
    for row in output:
        if row["dog_id"] not in assignment:
            # An invalid-only dog cannot leak and follows train by default.
            row["split"] = "train"
        else:
            row["split"] = assignment[row["dog_id"]]
    return output


def split_summary(rows, label_key="behavior_label"):
    summary = {}
    for split in SPLITS:
        selected = [r for r in rows if r.get("split") == split and r.get("valid") == "1"]
        summary[split] = {
            "dogs": len({r["dog_id"] for r in selected}),
            "clips": len(selected),
            "labels": dict(sorted(Counter(r[label_key] for r in selected).items())),
        }
    return summary
