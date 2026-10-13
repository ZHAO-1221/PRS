"""Run directly: python tests/test_keypoint_metrics.py (the functions are also pytest-style)."""
import numpy as np

from pet_recognition.evaluation.keypoint_metrics import (AP10K_SIGMAS, average_precision, keypoint_similarity,
                                                         paired_bootstrap, pck)


def test_perfect_prediction_scores_one():
    gt = np.random.default_rng(0).uniform(0, 500, (17, 2))
    gt[3] = np.nan  # an unlabeled point is ignored, not counted as wrong
    sim = keypoint_similarity(gt.copy(), gt, area=200 * 100)
    assert np.isnan(sim[3]) and np.nanmean(sim) == 1
    assert pck(np.where(np.isnan(gt[:, 0]), np.nan, 0.0), 0.05) == 1
    assert average_precision(np.ones(5), np.arange(5.0)).mean() == 1


def test_error_lowers_scores():
    gt = np.full((17, 2), 100.0)
    pred = gt.copy()
    pred[0, 0] += 30  # one eye 30 px off on a 200x100 box
    sim = keypoint_similarity(pred, gt, area=200 * 100)
    assert sim[0] < 0.01 and (sim[1:] == 1).all()
    distance = np.sqrt(((pred - gt) ** 2).sum(1)) / 200
    assert pck(distance, 0.1) == 16 / 17
    # the same 30 px costs a hip (large sigma) less than an eye (small sigma)
    assert AP10K_SIGMAS[11] > AP10K_SIGMAS[0]


def test_missing_detection_counts_every_labeled_point_as_a_miss():
    gt = np.full((17, 2), 100.0)
    gt[5] = np.nan
    pred = np.full((17, 2), np.inf)  # what tools/eval_pose.py predicts when the detector found no dog
    sim = keypoint_similarity(pred, gt, area=200 * 100)
    assert np.isnan(sim[5]) and np.nanmean(sim) == 0
    assert pck(np.sqrt(((pred - gt) ** 2).sum(1)) / 200, 0.1) == 0


def test_average_precision_counts_misses():
    oks = np.array([0.9, 0.9, 0.2, 0.2])
    aps = average_precision(oks, scores=np.array([4.0, 3.0, 2.0, 1.0]))
    assert np.isclose(aps[0], 0.5, atol=0.01)  # confident hits first: precision 1 up to recall 0.5
    assert aps[-1] == 0  # nothing reaches OKS 0.95


def test_paired_bootstrap_interval():
    assert np.allclose(paired_bootstrap(np.full(30, 0.1)), 0.1)
    mean, low, high = paired_bootstrap(np.array([0.2, -0.2] * 15))
    assert low < 0 < high


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("ok")
