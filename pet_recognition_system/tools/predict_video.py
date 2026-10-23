"""Predict dog behaviour and human-dog interaction timelines from one video.

Examples:
    python tools/predict_video.py input.mp4 --task both --data-root <data-root>
    python tools/predict_video.py skeleton_window.npz --task dog_behavior
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pet_recognition.inference.video_pipeline import (
    DEFAULT_MIN_WINDOW_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    DOG_BEHAVIOR_CLASSES,
    INTERACTION_CLASSES,
    PROJECT_ROOT,
    apply_predictions,
    detection_rates,
    export_timeline_clips,
    iter_video_windows,
    load_skeleton_record,
    probe_video,
    smooth_and_merge,
    write_results,
)


OFFICIAL_CLIP_MODEL = "openai/clip-vit-base-patch32"


def _clip_model_source(data_root: Path) -> str:
    """Prefer a complete local CLIP snapshot, otherwise use the official Hub model."""
    local = data_root / "models" / "clip-vit-base-patch32"
    weight_files = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    if (local / "config.json").exists() and any(
        (local / name).exists() for name in weight_files
    ):
        return str(local)
    return OFFICIAL_CLIP_MODEL


def _pose_stack(data_root: Path, device: str):
    models = data_root / "models"
    detector_path = models / "rtmdet-m_coco" / "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
    human_path = models / "rtmpose-m_coco_human" / "end2end.onnx"
    dog_path = models / "rtmpose-m_ap10k" / "end2end.onnx"
    missing = [str(path) for path in (detector_path, human_path, dog_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing pose pipeline weights: " + ", ".join(missing))
    try:
        from pet_recognition.models.rtmdet import RTMDet
        from pet_recognition.models.rtmpose_onnx import RTMPoseOnnx
        from pet_recognition.inference.skeleton import extract_skeletons
    except ImportError as error:
        raise RuntimeError(
            "raw-video inference requires Pillow, onnxruntime, mmengine, mmcv and mmdet; "
            "install the pose dependencies from README.md"
        ) from error
    detector = RTMDet(detector_path, device=device)
    poses = {"person": RTMPoseOnnx(human_path), "dog": RTMPoseOnnx(dog_path)}
    return detector, poses, extract_skeletons


def _records_from_video(video: Path, data_root: Path, task: str, device: str,
                        window_seconds: float, stride_seconds: float | None,
                        min_window_seconds: float, use_visual: bool = False):
    detector, poses, extract_skeletons = _pose_stack(data_root, device)
    visual_encoder = None
    if use_visual and task in ("interaction", "both"):
        from pet_recognition.inference.interaction_visual import InteractionVisualEncoder
        visual_encoder = InteractionVisualEncoder(device, _clip_model_source(data_root))
    streams = ("dog",) if task == "dog_behavior" else ("person", "dog")
    metadata = probe_video(video)
    records, windows = [], []
    for index, window in enumerate(iter_video_windows(
        video, window_seconds=window_seconds, stride_seconds=stride_seconds,
        min_window_seconds=min_window_seconds
    )):
        skeletons, boxes = extract_skeletons(window["frames"], detector, poses, streams)
        record = {"clip_id": f"window_{index:04d}", "skeletons": skeletons,
                  "boxes": boxes, "image_size": np.asarray((metadata["width"], metadata["height"])),
                  "fps": metadata["sample_fps"]}
        if visual_encoder is not None:
            record["interaction_visual_features"] = visual_encoder.encode(
                window["frames"], boxes
            )
        records.append(record)
        windows.append({"index": index, "start_sec": window["start_sec"],
                        "end_sec": window["end_sec"], "detection": detection_rates(record)})
        print(f"window {index + 1}: {window['start_sec']:.1f}-{window['end_sec']:.1f}s, "
              f"person={windows[-1]['detection']['person']:.0%}, "
              f"dog={windows[-1]['detection']['dog']:.0%}", flush=True)
    return records, windows, metadata


def _record_from_npz(path: Path):
    record = load_skeleton_record(path)
    duration = record["skeletons"].shape[1] / record["fps"]
    metadata = {"duration_sec": duration, "sample_fps": record["fps"],
                "width": int(record["image_size"][0]), "height": int(record["image_size"][1]),
                "source": "precomputed_skeleton"}
    windows = [{"index": 0, "start_sec": 0.0, "end_sec": duration,
                "detection": detection_rates(record)}]
    return [record], windows, metadata


def predict_video(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    task: str = "both",
    data_root: str | Path | None = None,
    model_root: str | Path | None = None,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    stride_seconds: float | None = None,
    min_window_seconds: float = DEFAULT_MIN_WINDOW_SECONDS,
    min_detection: float = 0.5,
    smoothing_radius: int = 1,
    device: str = "cpu",
    export_clips: bool = False,
) -> dict:
    """Run the complete inference pipeline for one video or skeleton file.

    This is the public Python entry point intended for use by an API or another
    application. It writes ``result.json`` and ``timeline.csv`` and returns
    their paths together with the merged timeline.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input does not exist: {input_path}")
    if task not in {"dog_behavior", "interaction", "both"}:
        raise ValueError(f"unsupported task: {task}")

    model_root = Path(model_root) if model_root else PROJECT_ROOT / "checkpoints" / "best_models"
    output = (Path(output_dir) if output_dir else
              PROJECT_ROOT / "outputs" / "predictions" / input_path.stem)
    if input_path.suffix.lower() == ".npz":
        records, windows, metadata = _record_from_npz(input_path)
    else:
        resolved_data_root = data_root or os.environ.get("PET_DATA_ROOT")
        if not resolved_data_root:
            raise ValueError("raw video requires data_root or PET_DATA_ROOT")
        records, windows, metadata = _records_from_video(
            input_path, Path(resolved_data_root), task, device,
            window_seconds, stride_seconds, min_window_seconds,
            use_visual=True,
        )
    if not records:
        raise RuntimeError("no usable video windows were produced")

    apply_predictions(
        windows, records, task=task, model_root=model_root,
        device=device, min_detection=min_detection,
    )
    timeline = {}
    if task in ("dog_behavior", "both"):
        dog_timeline = smooth_and_merge(
            windows, "dog_behavior", DOG_BEHAVIOR_CLASSES,
            metadata["duration_sec"], smoothing_radius,
        )
        if dog_timeline:
            timeline["dog_behavior"] = dog_timeline
    if task in ("interaction", "both"):
        timeline["interaction"] = smooth_and_merge(
            windows, "interaction", INTERACTION_CLASSES,
            metadata["duration_sec"], smoothing_radius,
        )

    result = {
        "input": str(input_path.resolve()),
        "task": task,
        "metadata": metadata,
        "configuration": {
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds or window_seconds,
            "min_detection": min_detection,
            "smoothing_radius": smoothing_radius,
        },
        "windows": windows,
        "timeline": timeline,
    }
    json_path, csv_path = write_results(output, result)
    if export_clips:
        if input_path.suffix.lower() == ".npz":
            raise ValueError("export_clips requires a video input")
        result["exported_clips"] = export_timeline_clips(input_path, output, timeline)
        json_path, csv_path = write_results(output, result)
    return {
        "result_json": str(json_path),
        "timeline_csv": str(csv_path),
        "timeline": timeline,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="MP4 video or a precomputed skeleton NPZ")
    parser.add_argument("--task", choices=("dog_behavior", "interaction", "both"), default="both")
    parser.add_argument("--data-root", type=Path, default=os.environ.get("PET_DATA_ROOT"),
                        help="folder containing models/; required for raw video")
    parser.add_argument("--model-root", type=Path,
                        default=PROJECT_ROOT / "checkpoints" / "best_models")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--stride-seconds", type=float)
    parser.add_argument("--min-window-seconds", type=float, default=DEFAULT_MIN_WINDOW_SECONDS)
    parser.add_argument("--min-detection", type=float, default=0.5)
    parser.add_argument("--smoothing-radius", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--export-clips", action="store_true")
    args = parser.parse_args()
    try:
        result = predict_video(
            args.input,
            args.output_dir,
            task=args.task,
            data_root=args.data_root,
            model_root=args.model_root,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            min_window_seconds=args.min_window_seconds,
            min_detection=args.min_detection,
            smoothing_radius=args.smoothing_radius,
            device=args.device,
            export_clips=args.export_clips,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
