"""Extract person and dog skeleton sequences from video clips or frame folders.

    python tools/extract_skeletons.py CLIP [CLIP ...] [--out-dir outputs/skeletons] [--video]

CLIP is a video file or a folder of frames (sorted .jpg, taken as 15 fps). Videos are resampled to 15 fps,
the rate the behavior clips were sliced at. For each clip, <out-dir>/<name>.npz holds
skeletons (2, T, 17, 3): stream 0 person (COCO keypoints), stream 1 dog (AP-10K keypoints), each point
x, y in pixels and a confidence in (0, 1]; a stream is all zeros in frames where its target was not
detected.
It also holds boxes (2, T, 5), fps and image_size (width, height). --video writes <name>_skeleton.mp4
with the skeletons drawn on, for checking by eye. Weights come from <data root>/models ($PET_DATA_ROOT).
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from pet_recognition.inference.skeleton import STREAMS, extract_skeletons
from pet_recognition.models.rtmdet import RTMDet
from pet_recognition.models.rtmpose_onnx import RTMPoseOnnx
from pet_recognition.visualization.skeleton import draw_skeletons

FPS = 15


def read_frames(clip):
    """BGR frames at FPS frames per second."""
    if clip.is_dir():
        # frames are named frame1.jpg, frame2.jpg ... so sort by the number, not the text
        paths = sorted(clip.glob("*.jpg"), key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))
        # imdecode instead of imread: imread cannot open non-ASCII paths on Windows
        return [cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR) for p in paths]
    video = cv2.VideoCapture(str(clip))
    source_fps = video.get(cv2.CAP_PROP_FPS)
    frames, index = [], 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        if int(index * FPS / source_fps) == len(frames):  # keep the first frame of every 1/FPS second
            frames.append(frame)
        index += 1
    return frames


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--data-root", default=os.environ.get("PET_DATA_ROOT"))
    parser.add_argument("--out-dir", type=Path, default=project / "outputs" / "skeletons")
    parser.add_argument("--video", action="store_true", help="also write a video with the skeletons drawn on")
    args = parser.parse_args()
    if not args.data_root:
        parser.error("set PET_DATA_ROOT or pass --data-root")
    models = Path(args.data_root) / "models"
    detector = RTMDet(models / "rtmdet-m_coco" / "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth")
    pose_models = {"person": RTMPoseOnnx(models / "rtmpose-m_coco_human" / "end2end.onnx"),
                   "dog": RTMPoseOnnx(models / "rtmpose-m_ap10k" / "end2end.onnx")}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for clip in args.clips:
        frames = read_frames(clip)
        assert frames, f"{clip}: no frames read"
        skeletons, boxes = extract_skeletons(frames, detector, pose_models)
        height, width = frames[0].shape[:2]
        np.savez_compressed(args.out_dir / f"{clip.stem}.npz", skeletons=skeletons, boxes=boxes, fps=FPS,
                            image_size=(width, height))
        found = ", ".join(f"{s} {np.mean(boxes[i, :, 4] > 0):.0%}" for i, s in enumerate(STREAMS))
        print(f"{clip.stem}: {len(frames)} frames, detected in {found}")
        if args.video:
            out = cv2.VideoWriter(str(args.out_dir / f"{clip.stem}_skeleton.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                  FPS, (width, height))
            for t, frame in enumerate(frames):
                out.write(draw_skeletons(frame, skeletons[:, t], boxes[:, t], STREAMS))
            out.release()


if __name__ == "__main__":
    main()
