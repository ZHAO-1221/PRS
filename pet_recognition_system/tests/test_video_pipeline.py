import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from pet_recognition.datasets.dog_behavior import DOG_BEHAVIOR_CLASSES
from pet_recognition.datasets.interaction import INTERACTION_CLASSES
from pet_recognition.inference.video_pipeline import (
    DogBehaviorPredictor,
    InteractionPredictor,
    detection_rates,
    iter_video_windows,
    smooth_and_merge,
)


PROJECT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT / "checkpoints" / "best_models"


def synthetic_record(frames=60):
    skeletons = np.zeros((2, frames, 17, 3), dtype=np.float32)
    boxes = np.zeros((2, frames, 5), dtype=np.float32)
    time = np.linspace(0, 2 * np.pi, frames, dtype=np.float32)

    human = np.asarray([
        (200, 80), (190, 75), (210, 75), (180, 80), (220, 80),
        (170, 150), (230, 150), (150, 220), (250, 220), (140, 290),
        (260, 290), (180, 300), (220, 300), (175, 400), (225, 400),
        (170, 500), (230, 500),
    ], dtype=np.float32)
    dog = np.asarray([
        (420, 240), (440, 240), (430, 260), (410, 300), (560, 330),
        (405, 320), (395, 370), (390, 430), (435, 320), (445, 370),
        (450, 430), (520, 350), (515, 400), (510, 450), (555, 350),
        (560, 400), (565, 450),
    ], dtype=np.float32)
    for index in range(frames):
        human_offset = np.asarray((4 * np.sin(time[index]), 0), dtype=np.float32)
        dog_offset = np.asarray((8 * index / max(frames - 1, 1),
                                 2 * np.sin(2 * time[index])), dtype=np.float32)
        skeletons[0, index, :, :2] = human + human_offset
        skeletons[1, index, :, :2] = dog + dog_offset
        skeletons[:, index, :, 2] = 0.9
        boxes[0, index] = (120 + human_offset[0], 50, 280 + human_offset[0], 520, 0.95)
        boxes[1, index] = (370 + dog_offset[0], 210 + dog_offset[1],
                           590 + dog_offset[0], 470 + dog_offset[1], 0.92)
    return {"clip_id": "synthetic", "skeletons": skeletons, "boxes": boxes,
            "image_size": np.asarray((640, 540), dtype=np.float32), "fps": 15.0}


def test_released_predictors_accept_one_unlabelled_window():
    record = synthetic_record()
    dog = DogBehaviorPredictor(MODEL_ROOT / "dog_behavior").predict([record])
    interaction = InteractionPredictor(MODEL_ROOT / "interaction").predict([record])
    assert dog.shape == (1, len(DOG_BEHAVIOR_CLASSES))
    assert interaction.shape == (1, len(INTERACTION_CLASSES))
    assert np.isfinite(dog).all() and np.isclose(dog.sum(), 1, atol=1e-5)
    assert np.isfinite(interaction).all() and np.isclose(interaction.sum(), 1, atol=1e-5)
    assert detection_rates(record) == {"person": 1.0, "dog": 1.0}


def test_released_interaction_predictor_accepts_visual_feature():
    record = synthetic_record()
    record["interaction_visual_features"] = np.zeros(4608, dtype=np.float32)
    probability = InteractionPredictor(MODEL_ROOT / "interaction").predict([record])
    assert probability.shape == (1, len(INTERACTION_CLASSES))
    assert np.isfinite(probability).all() and np.isclose(probability.sum(), 1, atol=1e-5)


def test_smoothing_produces_non_overlapping_merged_timeline():
    windows = []
    for index, values in enumerate(((0.8, 0.1, 0.05, 0.03, 0.02),
                                    (0.7, 0.2, 0.05, 0.03, 0.02),
                                    (0.1, 0.1, 0.1, 0.6, 0.1))):
        windows.append({"start_sec": index * 4.0, "end_sec": (index + 1) * 4.0,
                        "dog_behavior": {"status": "ok", "raw_probabilities": {
                            name: value for name, value in zip(DOG_BEHAVIOR_CLASSES, values)
                        }}})
    timeline = smooth_and_merge(
        windows, "dog_behavior", DOG_BEHAVIOR_CLASSES, 12.0, radius=0
    )
    assert [(row["start_sec"], row["end_sec"], row["label"]) for row in timeline] == [
        (0.0, 8.0, "walking_run"), (8.0, 12.0, "lying_sleeping")
    ]


def test_video_decoder_uses_one_window_for_short_video(tmp_path):
    path = tmp_path / "short.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for index in range(60):
        writer.write(np.full((48, 64, 3), index % 255, dtype=np.uint8))
    writer.release()
    windows = list(iter_video_windows(path))
    assert len(windows) == 1
    assert windows[0]["start_sec"] == 0
    assert 5.8 <= windows[0]["end_sec"] <= 6.1
    assert len(windows[0]["frames"]) == 60


def test_cli_predicts_precomputed_skeleton_with_released_models(tmp_path):
    record = synthetic_record()
    skeleton_path = tmp_path / "window.npz"
    np.savez_compressed(skeleton_path, skeletons=record["skeletons"], boxes=record["boxes"],
                        image_size=record["image_size"], fps=record["fps"])
    output = tmp_path / "prediction"
    subprocess.run([
        sys.executable, str(PROJECT / "tools" / "predict_video.py"), str(skeleton_path),
        "--task", "both", "--output-dir", str(output), "--smoothing-radius", "0",
    ], cwd=PROJECT, check=True, capture_output=True, text=True)
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert (output / "timeline.csv").exists()
    assert "dog_behavior" not in result["timeline"]
    assert result["timeline"]["interaction"][0]["label"] in INTERACTION_CLASSES


def test_public_predict_video_function(tmp_path):
    from tools.predict_video import predict_video

    record = synthetic_record()
    skeleton_path = tmp_path / "window.npz"
    np.savez_compressed(skeleton_path, skeletons=record["skeletons"], boxes=record["boxes"],
                        image_size=record["image_size"], fps=record["fps"])
    output = tmp_path / "function_prediction"
    response = predict_video(
        skeleton_path, output, task="both", smoothing_radius=0,
    )
    assert Path(response["result_json"]).exists()
    assert Path(response["timeline_csv"]).exists()
    assert "dog_behavior" not in response["timeline"]
    assert response["timeline"]["interaction"][0]["label"] in INTERACTION_CLASSES


def test_raw_video_orchestration_with_replaceable_pose_stack(tmp_path, monkeypatch):
    from tools import predict_video

    path = tmp_path / "long.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for index in range(100):
        writer.write(np.full((48, 64, 3), index % 255, dtype=np.uint8))
    writer.release()

    def fake_extract(frames, detector, poses, streams):
        record = synthetic_record(len(frames))
        return record["skeletons"], record["boxes"]

    monkeypatch.setattr(predict_video, "_pose_stack",
                        lambda data_root, device: (object(), {}, fake_extract))
    records, windows, metadata = predict_video._records_from_video(
        path, tmp_path, "both", "cpu", 4.0, None, 1.5
    )
    assert len(records) == len(windows) == 3
    assert [(row["start_sec"], round(row["end_sec"], 1)) for row in windows] == [
        (0.0, 4.0), (4.0, 8.0), (8.0, 10.0)
    ]
    assert metadata["duration_sec"] == 10.0
