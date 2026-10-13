"""Keypoint metrics: OKS, COCO-style OKS AP, PCK and a paired bootstrap for comparing two models."""
import numpy as np

# AP-10K sigmas, read from the official RTMPose AP-10K checkpoint's dataset_meta
AP10K_SIGMAS = np.array([0.025, 0.025, 0.026, 0.035, 0.035, 0.079, 0.072, 0.062, 0.079,
                         0.072, 0.062, 0.107, 0.087, 0.089, 0.107, 0.087, 0.089])
OKS_THRESHOLDS = np.arange(0.5, 0.96, 0.05)


def keypoint_similarity(pred_xy, gt_xy, area, sigmas=AP10K_SIGMAS):
    """Per-keypoint OKS term; NaN where the ground truth is unlabeled (NaN). An image's OKS is the nanmean."""
    d2 = ((pred_xy - gt_xy) ** 2).sum(-1)
    return np.exp(-d2 / (2 * area * (2 * sigmas) ** 2))


def pck(normalized_distance, threshold):
    """Share of labeled keypoints whose error, as a fraction of the box's long side, is within the threshold."""
    labeled = ~np.isnan(normalized_distance)
    return (normalized_distance[labeled] <= threshold).mean()


def average_precision(oks, scores):
    """AP per OKS threshold. Assumes one ground-truth animal and one prediction per image."""
    ranked = oks[np.argsort(-scores)]
    aps = []
    for t in OKS_THRESHOLDS:
        tp = np.cumsum(ranked >= t)
        recall, precision = tp / len(ranked), tp / np.arange(1, len(ranked) + 1)
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
        aps.append(np.where(idx < len(ranked), precision[np.minimum(idx, len(ranked) - 1)], 0).mean())
    return np.array(aps)


def paired_bootstrap(differences, resamples=10000, seed=0):
    """Mean of per-image differences with a 95% bootstrap confidence interval (images resampled)."""
    means = np.random.default_rng(seed).choice(differences, (resamples, len(differences))).mean(1)
    return differences.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)
