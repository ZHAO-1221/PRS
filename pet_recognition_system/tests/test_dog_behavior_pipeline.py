"""Tests for the dog-behaviour skeleton pipeline."""
import csv

import numpy as np
import pytest

from pet_recognition.datasets.group_split import apply_group_assignment, balanced_group_assignment
from pet_recognition.evaluation.classification_metrics import classification_metrics
from pet_recognition.models.ap10k_graph import spatial_adjacency
from pet_recognition.preprocessing.skeleton import (
    extract_global_motion,
    extract_box_motion,
    interpolate_missing,
    normalize_dog_skeleton,
    temporal_sample,
)


def test_temporal_sampling_and_interpolation():
    points = np.zeros((4, 17, 3), np.float32)
    points[0, :, :2], points[0, :, 2] = 10, 1
    points[3, :, :2], points[3, :, 2] = 40, 1
    filled = interpolate_missing(points)
    assert np.allclose(filled[1, :, :2], 20)
    assert (filled[1, :, 2] == 0).all()
    sampled, mask = temporal_sample(filled, 6)
    assert sampled.shape == (6, 17, 3)
    assert mask.tolist() == [1, 1, 1, 1, 0, 0]


def test_normalization_is_translation_and_scale_invariant():
    rng = np.random.default_rng(2)
    points = rng.uniform(100, 500, (5, 17, 3)).astype(np.float32)
    points[..., 2] = 1
    a = normalize_dog_skeleton(points, np.array([1920, 1080]))
    moved = points.copy()
    moved[..., :2] = moved[..., :2] * 2 + np.array([300, -40])
    b = normalize_dog_skeleton(moved, np.array([3840, 2160]))
    assert np.allclose(a[..., :2], b[..., :2], atol=1e-5)


def test_global_motion_preserves_translation():
    points = np.zeros((4, 17, 3), np.float32)
    points[..., 0] = np.arange(4)[:, None] * 64 + np.arange(17)[None, :]
    points[..., 1] = 100 + np.arange(17)[None, :]
    points[..., 2] = 1
    motion = extract_global_motion(points, np.array([640, 480]))
    assert motion.shape == (4, 6)
    assert motion[-1, 0] > 0.2
    assert motion[1:, 2].mean() > 0

    boxes = np.array([[0, 0, 100, 50, 1], [64, 0, 164, 50, 1],
                      [128, 0, 228, 50, 1], [192, 0, 292, 50, 1]], np.float32)
    box_motion = extract_box_motion(boxes, np.array([640, 480]))
    assert box_motion[-1, 0] > 0.2
    assert box_motion[1:, 2].mean() > 0


def test_ap10k_graph_has_three_partitions():
    adjacency = spatial_adjacency()
    assert adjacency.shape == (3, 17, 17)
    assert np.allclose(np.diag(adjacency[0]), 1)
    assert np.isfinite(adjacency).all() and (adjacency >= 0).all()


def test_group_split_has_no_dog_leakage_and_every_class_in_each_split():
    labels = ["walking_run", "standing", "sitting", "lying_sleeping", "eating_drinking"]
    rows = []
    for dog in range(18):
        for label in labels:
            rows.append({"dog_id": f"d{dog:03d}", "behavior_label": label, "valid": "1", "split": ""})
    assignment = balanced_group_assignment(rows, seed=7, trials=500)
    output = apply_group_assignment(rows, assignment)
    for split in ("train", "val", "test"):
        selected = [row for row in output if row["split"] == split]
        assert {row["behavior_label"] for row in selected} == set(labels)
    for dog in {row["dog_id"] for row in output}:
        assert len({row["split"] for row in output if row["dog_id"] == dog}) == 1


def test_macro_f1_weights_classes_equally():
    report = classification_metrics([0, 0, 1, 1], [0, 0, 0, 1], ["a", "b"])
    assert report["accuracy"] == 0.75
    assert np.isclose(report["macro_f1"], (0.8 + 2 / 3) / 2)


def test_dataset_and_models(tmp_path):
    torch = pytest.importorskip("torch")
    from pet_recognition.datasets.dog_behavior import DogBehaviorSkeletonDataset
    from pet_recognition.models.skeleton_lstm import DogBehaviorLSTM
    from pet_recognition.models.stgcn import DogBehaviorSTGCN
    from pet_recognition.models.skeleton_transformer import DogBehaviorTransformer
    from pet_recognition.models.skeleton_tcn import (
        DogBehaviorMotionAuxTCN,
        DogBehaviorMotionSummaryTCN,
        DogBehaviorTCN,
    )

    skeleton_dir = tmp_path / "skeletons"
    skeleton_dir.mkdir()
    points = np.zeros((2, 40, 17, 3), np.float32)
    points[1, ..., 0] = np.linspace(100, 500, 17)
    points[1, ..., 1] = np.linspace(50, 300, 17)
    points[1, ..., 2] = 0.9
    np.savez(skeleton_dir / "dog_d001_v01_c001.npz", skeletons=points,
             boxes=np.zeros((2, 40, 5), np.float32), fps=np.int32(15), image_size=np.array([640, 480]))
    manifest = tmp_path / "clips.csv"
    fields = ["clip_id", "dog_id", "behavior_label", "split", "valid"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerow(dict(clip_id="d001_v01_c001", dog_id="d001", behavior_label="sitting", split="train", valid="1"))
    dataset = DogBehaviorSkeletonDataset(manifest, skeleton_dir, "train", augment=False)
    item = dataset[0]
    assert item["keypoints"].shape == (3, 32, 17, 1)
    assert item["global_motion"].shape == (32, 6)
    x, mask, motion = (item["keypoints"][None], item["frame_mask"][None],
                       item["global_motion"][None])
    for model in (DogBehaviorSTGCN(), DogBehaviorLSTM(), DogBehaviorTransformer(),
                  DogBehaviorTCN(), DogBehaviorMotionSummaryTCN(),
                  DogBehaviorMotionAuxTCN()):
        output = model(x, mask, motion)
        assert output.shape == (1, 5)
        output.sum().backward()


def test_interaction_model_forward():
    torch = pytest.importorskip("torch")
    from pet_recognition.models.interaction_stgcn import HumanDogInteractionSTGCN
    human = torch.randn(2, 3, 32, 17, 1)
    dog = torch.randn(2, 3, 32, 17, 1)
    relation = torch.randn(2, 32, 48)
    mask = torch.ones(2, 32)
    output = HumanDogInteractionSTGCN()(human, dog, relation, mask)
    assert output.shape == (2, 4)
    from pet_recognition.datasets.interaction import summarize_relation_features
    summary = summarize_relation_features(relation[0].numpy(), mask[0].numpy())
    assert summary.shape == (48 * 7 + 16,)
