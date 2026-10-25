"""End-to-end windowed inference for dog behaviour and human-dog interaction."""
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import cv2
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset

from pet_recognition.datasets.dog_behavior import DOG_BEHAVIOR_CLASSES
from pet_recognition.datasets.interaction import (
    INTERACTION_CLASSES,
    summarize_relation_features,
)
from pet_recognition.preprocessing.skeleton import (
    apply_temporal_indices,
    extract_box_motion,
    extract_global_motion,
    extract_relation_features,
    interpolate_missing,
    normalize_dog_skeleton,
    temporal_indices,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The released estimators were saved with NumPy 2, whose internal pickle path
# changed from ``numpy.core`` to ``numpy._core``.  NumPy 1.26 is retained here
# for compatibility with the OpenMMLab/PyTorch runtime, so expose the new path
# while loading trusted project-owned joblib files.
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.numeric"] = np.core.numeric
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

# These functions define the exact engineered features used by the frozen
# release. Keeping inference on the same implementation prevents train/serve
# feature drift.
from cross_validate_dog_joint_specialist import combined_features  # noqa: E402
from train_dog_eating_specialist import dataset_features as eating_features  # noqa: E402
from train_dog_play_specialist import tcn_probabilities  # noqa: E402


TARGET_FPS = 15.0
TARGET_FRAMES = 32
DEFAULT_WINDOW_SECONDS = 4.0
DEFAULT_MIN_WINDOW_SECONDS = 1.5
SHORT_VIDEO_LIMIT_SECONDS = 8.0


def _validate_record(record: dict) -> None:
    skeletons = np.asarray(record["skeletons"])
    boxes = np.asarray(record["boxes"])
    image_size = np.asarray(record["image_size"])
    if skeletons.ndim != 4 or skeletons.shape[0] != 2 or skeletons.shape[2:] != (17, 3):
        raise ValueError(f"expected skeletons (2,T,17,3), got {skeletons.shape}")
    if boxes.shape != (2, skeletons.shape[1], 5):
        raise ValueError(f"expected boxes (2,T,5), got {boxes.shape}")
    if image_size.shape != (2,) or np.any(image_size <= 0):
        raise ValueError(f"invalid image_size {image_size}")


class DogWindowDataset(Dataset):
    """Apply the training-time dog preprocessing to in-memory windows."""

    def __init__(self, records: Iterable[dict], target_frames: int = TARGET_FRAMES):
        self.records = list(records)
        self.target_frames = target_frames
        self.rows = [{"clip_id": value.get("clip_id", f"window_{index:04d}"),
                      "dog_id": "inference"}
                     for index, value in enumerate(self.records)]
        for record in self.records:
            _validate_record(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        skeletons = np.asarray(record["skeletons"], dtype=np.float32)
        boxes = np.asarray(record["boxes"], dtype=np.float32)
        image_size = np.asarray(record["image_size"], dtype=np.float32)
        dog = interpolate_missing(skeletons[1])
        if boxes.shape[:2] == skeletons.shape[:2]:
            global_motion = extract_box_motion(boxes[1], image_size)
        else:
            global_motion = extract_global_motion(dog, image_size)
        dog = normalize_dog_skeleton(dog, image_size)
        indices, frame_mask = temporal_indices(len(dog), self.target_frames, training=False)
        dog = apply_temporal_indices(dog, indices, frame_mask, self.target_frames)
        global_motion = apply_temporal_indices(
            global_motion, indices, frame_mask, self.target_frames
        )
        data = np.ascontiguousarray(dog.transpose(2, 0, 1)[..., None])
        return {
            "keypoints": torch.from_numpy(data),
            "frame_mask": torch.from_numpy(frame_mask),
            "global_motion": torch.from_numpy(np.ascontiguousarray(global_motion)),
            # Feature helpers expect a label; it is ignored during inference.
            "label": torch.tensor(0, dtype=torch.long),
            "clip_id": self.rows[index]["clip_id"],
            "dog_id": "inference",
        }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def _apply_pair(logits: np.ndarray, features: np.ndarray, artifact: dict) -> np.ndarray:
    index = {name: position for position, name in enumerate(DOG_BEHAVIOR_CLASSES)}
    probability = artifact["model"].predict_proba(features)[:, 1]
    decision = np.log((probability + 0.02) / (1 - probability + 0.02))
    positive, negative = index[artifact["positive"]], index[artifact["negative"]]
    if artifact["gate"] == "pair_top":
        ranking = np.argsort(logits, axis=1)[:, -2:]
        enabled = np.logical_and((ranking == positive).any(1),
                                 (ranking == negative).any(1))
    elif artifact["gate"] == "close":
        enabled = np.abs(logits[:, positive] - logits[:, negative]) <= artifact["margin"]
    else:
        enabled = np.ones(len(logits), dtype=bool)
    logits[enabled, positive] += artifact["alpha"] * decision[enabled]
    logits[enabled, negative] -= artifact["alpha"] * decision[enabled]
    return logits


class DogBehaviorPredictor:
    """The single published five-fold dog-behaviour ensemble."""

    def __init__(self, model_root: str | Path, device: str = "cpu"):
        self.root = Path(model_root)
        self.device = device
        self.backbone = self.root / "backbone"
        self.fusion = self.root / "fusion"
        required = [self.backbone / f"fold_{fold}" / "best.pt" for fold in range(5)]
        required += [self.fusion / f"fold_{fold}_multiclass.joblib" for fold in range(5)]
        required += [self.fusion / f"stage_{stage}_fold_{fold}.joblib"
                     for stage in (1, 2) for fold in range(5)]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing dog model files; first: {missing[:3]}")

    def predict(self, records: Iterable[dict]) -> np.ndarray:
        dataset = DogWindowDataset(records)
        if not len(dataset):
            return np.empty((0, len(DOG_BEHAVIOR_CLASSES)), dtype=np.float32)
        features, _ = combined_features(dataset, "head_duration_v1")
        eating_x, _ = eating_features(dataset, include_fixed_region=False)
        probabilities = []
        for fold in range(5):
            base = tcn_probabilities(
                self.backbone / f"fold_{fold}" / "best.pt", dataset, self.device
            )
            logits = np.log(base + 1e-8)
            multiclass = joblib.load(self.fusion / f"fold_{fold}_multiclass.joblib")
            logits += multiclass["alpha"] * np.log(
                multiclass["model"].predict_proba(features) + 1e-5
            )
            for stage in (1, 2):
                artifact = joblib.load(self.fusion / f"stage_{stage}_fold_{fold}.joblib")
                logits = _apply_pair(logits, features, artifact)
            duration = self.fusion / f"fold_{fold}_eating_duration.joblib"
            if duration.exists():
                artifact = joblib.load(duration)
                probability = artifact["model"].predict_proba(eating_x)[:, 1]
                decision = np.log((probability + 0.02) / (1 - probability + 0.02))
                eating = DOG_BEHAVIOR_CLASSES.index("eating_drinking")
                other = np.max(np.delete(logits, eating, axis=1), axis=1)
                enabled = (np.abs(logits[:, eating] - other) <= artifact["margin"]
                           if artifact["gate"] == "close" else np.ones(len(logits), bool))
                logits[enabled, eating] += artifact["alpha"] * decision[enabled]
            probabilities.append(_softmax(logits))
        return np.mean(probabilities, axis=0).astype(np.float32)


def interaction_feature_vector(record: dict) -> np.ndarray:
    """Reproduce the deterministic interaction preprocessing for one window."""
    _validate_record(record)
    skeletons = np.asarray(record["skeletons"], dtype=np.float32).copy()
    boxes = np.asarray(record["boxes"], dtype=np.float32)
    image_size = np.asarray(record["image_size"], dtype=np.float32)
    relation_human, relation_dog = skeletons[0].copy(), skeletons[1].copy()
    relation_human[relation_human[..., 2] < 0.2] = 0
    relation_dog[relation_dog[..., 2] < 0.2] = 0
    relation = extract_relation_features(
        interpolate_missing(relation_human),
        interpolate_missing(relation_dog),
        boxes,
        image_size,
    )
    indices, mask = temporal_indices(len(relation), TARGET_FRAMES, training=False)
    relation = apply_temporal_indices(relation, indices, mask, TARGET_FRAMES)
    return summarize_relation_features(relation, mask)


class InteractionPredictor:
    """Published relation ensemble with an optional frozen-CLIP visual branch."""

    def __init__(self, model_root: str | Path):
        self.root = Path(model_root)
        base_root = self.root / "checkpoints" if (self.root / "checkpoints").exists() else self.root
        paths = sorted(base_root.glob("fold_*.joblib"))
        if len(paths) != 5:
            raise FileNotFoundError(f"expected 5 interaction folds in {base_root}, found {len(paths)}")
        self.models = [joblib.load(path)["model"] for path in paths]
        release_root = base_root.parent if base_root.name == "checkpoints" else self.root
        rgb_paths = sorted((release_root / "rgb").glob("fold_*.joblib"))
        self.rgb_models = [joblib.load(path)["model"] for path in rgb_paths]
        config_path = release_root / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        self.visual_weight = float(config.get("visual_fusion_weight", 0.0))

    def predict(self, records: Iterable[dict]) -> np.ndarray:
        records = list(records)
        if not records:
            return np.empty((0, len(INTERACTION_CLASSES)), dtype=np.float32)
        features = np.stack([interaction_feature_vector(record) for record in records])
        skeleton = np.mean([model.predict_proba(features) for model in self.models], axis=0)
        if len(self.rgb_models) == 5 and self.visual_weight > 0 and all(
            "interaction_visual_features" in record for record in records
        ):
            visual = np.stack([record["interaction_visual_features"] for record in records])
            rgb = np.mean([model.predict_proba(visual) for model in self.rgb_models], axis=0)
            logits = np.log(skeleton + 1e-6) + self.visual_weight * np.log(rgb + 1e-6)
            return _softmax(logits).astype(np.float32)
        return skeleton.astype(np.float32)


def load_skeleton_record(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        record = {
            "clip_id": path.stem,
            "skeletons": archive["skeletons"].astype(np.float32),
            "boxes": archive["boxes"].astype(np.float32),
            "image_size": archive["image_size"].astype(np.float32),
            "fps": float(archive["fps"]) if "fps" in archive else TARGET_FPS,
        }
    _validate_record(record)
    return record


def probe_video(path: str | Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"invalid video metadata: fps={fps}, frames={frames}, size={width}x{height}")
    return {"source_fps": fps, "source_frames": frames, "duration_sec": frames / fps,
            "width": width, "height": height, "sample_fps": min(TARGET_FPS, fps)}


def iter_video_windows(path: str | Path, *, window_seconds: float = DEFAULT_WINDOW_SECONDS,
                       stride_seconds: float | None = None,
                       min_window_seconds: float = DEFAULT_MIN_WINDOW_SECONDS):
    """Decode one video lazily and yield uniformly sampled BGR frame windows."""
    metadata = probe_video(path)
    if window_seconds <= 0 or min_window_seconds <= 0:
        raise ValueError("window lengths must be positive")
    stride_seconds = window_seconds if stride_seconds is None else stride_seconds
    if stride_seconds <= 0 or stride_seconds > window_seconds:
        raise ValueError("stride_seconds must be in (0, window_seconds]")
    if metadata["duration_sec"] <= SHORT_VIDEO_LIMIT_SECONDS:
        window_seconds = metadata["duration_sec"]
        stride_seconds = window_seconds
    sample_fps = metadata["sample_fps"]
    window_frames = max(1, int(round(window_seconds * sample_fps)))
    stride_frames = max(1, int(round(stride_seconds * sample_fps)))
    minimum_frames = max(1, int(round(min_window_seconds * sample_fps)))
    capture = cv2.VideoCapture(str(path))
    buffer: list[np.ndarray] = []
    buffer_start = 0
    sampled = 0
    next_sample_time = 0.0
    last_yield_end = 0
    source_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            time_sec = source_index / metadata["source_fps"]
            source_index += 1
            if time_sec + 1e-9 < next_sample_time:
                continue
            buffer.append(frame)
            sampled += 1
            next_sample_time += 1.0 / sample_fps
            if len(buffer) >= window_frames:
                end = buffer_start + window_frames
                yield {"frames": list(buffer[:window_frames]),
                       "start_sec": buffer_start / sample_fps,
                       "end_sec": min(end / sample_fps, metadata["duration_sec"])}
                last_yield_end = end
                del buffer[:stride_frames]
                buffer_start += stride_frames
        end = buffer_start + len(buffer)
        if len(buffer) >= minimum_frames and end > last_yield_end:
            yield {"frames": list(buffer),
                   "start_sec": buffer_start / sample_fps,
                   "end_sec": min(end / sample_fps, metadata["duration_sec"])}
    finally:
        capture.release()


def detection_rates(record: dict) -> dict[str, float]:
    boxes = np.asarray(record["boxes"])
    return {"person": float(np.mean(boxes[0, :, 4] > 0)),
            "dog": float(np.mean(boxes[1, :, 4] > 0))}


def _probability_dict(classes: tuple[str, ...], values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(classes, values)}


def apply_predictions(windows: list[dict], records: list[dict], *, task: str,
                      model_root: str | Path, device: str = "cpu",
                      min_detection: float = 0.5, progress=None) -> None:
    """Attach raw class probabilities to every sufficiently detected window."""
    report = progress or (lambda stage, done, total: None)
    rates = [detection_rates(record) for record in records]
    if task in ("dog_behavior", "both"):
        report("behaviour", 0, 1)
        valid = [index for index, values in enumerate(rates)
                 if values["dog"] >= min_detection
                 and (task != "both" or values["person"] < min_detection)]
        if valid:
            predictor = DogBehaviorPredictor(Path(model_root) / "dog_behavior", device)
            probabilities = predictor.predict([records[index] for index in valid])
            for index, values in zip(valid, probabilities):
                windows[index]["dog_behavior"] = {
                    "status": "ok", "raw_probabilities": _probability_dict(DOG_BEHAVIOR_CLASSES, values)
                }
        for index in set(range(len(records))) - set(valid):
            status = ("suppressed_by_interaction" if task == "both"
                      and all(rates[index][name] >= min_detection for name in ("person", "dog"))
                      else "insufficient_detection")
            windows[index]["dog_behavior"] = {"status": status}
    if task in ("interaction", "both"):
        report("interaction", 0, 1)
        valid = [index for index, values in enumerate(rates)
                 if all(value >= min_detection for value in values.values())]
        if valid:
            predictor = InteractionPredictor(Path(model_root) / "interaction")
            probabilities = predictor.predict([records[index] for index in valid])
            for index, values in zip(valid, probabilities):
                windows[index]["interaction"] = {
                    "status": "ok", "raw_probabilities": _probability_dict(INTERACTION_CLASSES, values)
                }
        for index in set(range(len(records))) - set(valid):
            windows[index]["interaction"] = {"status": "insufficient_detection"}


def smooth_and_merge(windows: list[dict], task: str, classes: tuple[str, ...],
                     duration_sec: float, radius: int = 1) -> list[dict]:
    """Smooth adjacent probabilities and convert windows into a non-overlapping timeline."""
    radius = max(int(radius), 0)
    for index, window in enumerate(windows):
        value = window.get(task, {"status": "insufficient_detection"})
        if value.get("status") == "suppressed_by_interaction":
            continue
        if value.get("status") != "ok":
            value.update({"label": "insufficient_detection", "confidence": 0.0})
            continue
        neighbors = []
        for other in windows[max(0, index - radius):index + radius + 1]:
            candidate = other.get(task, {})
            if candidate.get("status") == "ok":
                neighbors.append([candidate["raw_probabilities"][name] for name in classes])
        probability = np.mean(neighbors, axis=0)
        selected = int(np.argmax(probability))
        value["label"] = classes[selected]
        value["confidence"] = float(probability[selected])
        value["probabilities"] = _probability_dict(classes, probability)

    centers = [(window["start_sec"] + window["end_sec"]) / 2 for window in windows]
    boundaries = [0.0]
    for index in range(len(centers) - 1):
        current, following = windows[index], windows[index + 1]
        # Non-overlapping default windows change label at the next window's
        # actual start. Overlapping windows use the midpoint of their centres.
        boundary = (following["start_sec"] if following["start_sec"] >= current["end_sec"] - 1e-6
                    else (centers[index] + centers[index + 1]) / 2)
        boundaries.append(float(boundary))
    boundaries.append(float(duration_sec))
    segments: list[dict] = []
    for index, window in enumerate(windows):
        value = window.get(task, {"label": "insufficient_detection", "confidence": 0.0})
        if value.get("status") == "suppressed_by_interaction":
            continue
        current = {"start_sec": round(float(boundaries[index]), 3),
                   "end_sec": round(float(boundaries[index + 1]), 3),
                   "label": value["label"], "confidence": float(value["confidence"]),
                   "windows": 1}
        if segments and segments[-1]["label"] == current["label"]:
            previous = segments[-1]
            total = previous["windows"] + 1
            previous["confidence"] = (
                previous["confidence"] * previous["windows"] + current["confidence"]
            ) / total
            previous["windows"] = total
            previous["end_sec"] = current["end_sec"]
        else:
            segments.append(current)
    for segment in segments:
        segment["confidence"] = round(segment["confidence"], 6)
    return segments


def write_results(output_dir: str | Path, result: dict) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "result.json"
    csv_path = output / "timeline.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("task", "start_sec", "end_sec", "label", "confidence", "windows"))
        for task, segments in result["timeline"].items():
            for segment in segments:
                writer.writerow((task, segment["start_sec"], segment["end_sec"],
                                 segment["label"], segment["confidence"], segment["windows"]))
    return json_path, csv_path


def export_timeline_clips(video: str | Path, output_dir: str | Path, timeline: dict) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            ffmpeg = None
    if not ffmpeg or not Path(ffmpeg).exists():
        raise RuntimeError("--export-clips requires ffmpeg or the imageio-ffmpeg package")
    output = Path(output_dir).resolve()
    root = output / "clips"
    # This directory contains generated artifacts only. Recreate it so a second
    # run cannot leave clips from tasks that are now intentionally suppressed.
    if root.exists():
        shutil.rmtree(root)
    exported = []
    for task, segments in timeline.items():
        task_dir = root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        for index, segment in enumerate(segments, 1):
            label = re.sub(r"[^A-Za-z0-9_-]+", "_", segment["label"])
            target = task_dir / f"{index:03d}_{segment['start_sec']:.3f}-{segment['end_sec']:.3f}_{label}.mp4"
            command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                       "-ss", str(segment["start_sec"]), "-to", str(segment["end_sec"]),
                       "-i", str(video), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target)]
            subprocess.run(command, check=True)
            exported.append(str(target))
    return exported
