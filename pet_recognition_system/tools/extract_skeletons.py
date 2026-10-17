"""Extract person and dog skeleton sequences from video clips or frame folders.

    python tools/extract_skeletons.py CLIP [CLIP ...] [--out-dir outputs/skeletons] [--prefix dog_] [--video]
        [--skip-existing] [--expect person dog] [--max-low 5] [--low-score 0]

CLIP is a video file or a folder of frames (sorted .jpg, taken as 15 fps). Videos are resampled to 15 fps,
the rate the behavior clips were sliced at. For each clip, <out-dir>/<name>.npz holds
skeletons (2, T, 17, 3): stream 0 person (COCO keypoints), stream 1 dog (AP-10K keypoints), each point
x, y in pixels and a confidence in (0, 1]; a stream is all zeros in frames where its target was not
detected.
It also holds boxes (2, T, 5), fps and image_size (width, height). --video writes <name>_skeleton.mp4
with the skeletons drawn on, for checking by eye. Weights come from <data root>/models ($PET_DATA_ROOT).

Which streams are extracted follows the clip's folder: clips in a folder named dog_behavior only get the dog
stream (the person stream stays all zeros), clips in interaction get both; --expect overrides this. For batches: --prefix is put before every output
name; --skip-existing keeps clips already exported, so an interrupted batch can be resumed; a clip that fails
is reported and skipped. A clip where one of its extracted targets is detected in fewer than 80% of its frames
counts as low detection; with --max-low N the batch stops once more than N clips are low (exit code 3), and
the summary lists them. --low-score lowers the score a weak box needs to continue a track (default 0.1); 0 takes
any box RTMDet puts where the dog was, for a clip where it finds the dog but never scores it well (night
infrared d025_v24_c002, scored 0.05-0.3 as a cat). Run such clips on their own.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from pet_recognition.inference.skeleton import STREAMS, extract_skeletons
from pet_recognition.models import rtmdet
from pet_recognition.models.rtmdet import RTMDet
from pet_recognition.models.rtmpose_onnx import RTMPoseOnnx
from pet_recognition.visualization.skeleton import draw_skeletons

FPS = 15
LOW_DETECTION = 0.8  # share of frames an expected target must be found in
FOLDER_STREAMS = {"dog_behavior": ("dog",), "interaction": ("person", "dog")}  # the targets each data folder shows


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
    parser.add_argument("--prefix", default="", help="put before every output file name, e.g. dog_")
    parser.add_argument("--skip-existing", action="store_true", help="keep clips whose .npz already exists")
    parser.add_argument("--expect", nargs="+", choices=STREAMS,
                        help="targets the clips show, overriding the folder rule; other streams stay all zeros")
    parser.add_argument("--max-low", type=int, help="stop after more than this many low-detection clips")
    parser.add_argument("--low-score", type=float, default=rtmdet.FOLLOW_LOW_SCORE,
                        help="score a weak box overlapping the followed one needs; 0 trusts every box")
    args = parser.parse_args()
    if not args.data_root:
        parser.error("set PET_DATA_ROOT or pass --data-root")
    rtmdet.FOLLOW_LOW_SCORE = args.low_score  # read by choose() and follow() at call time
    models = Path(args.data_root) / "models"
    detector = RTMDet(models / "rtmdet-m_coco" / "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth")
    pose_models = {"person": RTMPoseOnnx(models / "rtmpose-m_coco_human" / "end2end.onnx"),
                   "dog": RTMPoseOnnx(models / "rtmpose-m_ap10k" / "end2end.onnx")}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    low, failed, stopped = [], [], False
    for clip in args.clips:
        name = f"{args.prefix}{clip.stem}"
        streams = args.expect or FOLDER_STREAMS.get(clip.parent.name, STREAMS)
        npz = args.out_dir / f"{name}.npz"
        try:
            if args.skip_existing and npz.exists():
                boxes, status = np.load(npz)["boxes"], "already exported"
            else:
                frames = read_frames(clip)
                if not frames:
                    raise ValueError("no frames read")
                skeletons, boxes = extract_skeletons(frames, detector, pose_models, streams)
                height, width = frames[0].shape[:2]
                np.savez_compressed(npz, skeletons=skeletons, boxes=boxes, fps=FPS, image_size=(width, height))
                status = f"{len(frames)} frames"
                if args.video:
                    out = cv2.VideoWriter(str(args.out_dir / f"{name}_skeleton.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                          FPS, (width, height))
                    for t, frame in enumerate(frames):
                        out.write(draw_skeletons(frame, skeletons[:, t], boxes[:, t], STREAMS))
                    out.release()
        except Exception as error:  # one broken video must not end a batch of hundreds
            failed.append(clip.name)
            print(f"{name}: FAILED ({error})", flush=True)
            continue
        found = {s: float(np.mean(boxes[i, :, 4] > 0)) for i, s in enumerate(STREAMS)}
        print(f"{name}: {status}, detected in " + ", ".join(f"{s} {v:.0%}" for s, v in found.items()), flush=True)
        if any(found[s] < LOW_DETECTION for s in streams):
            low.append(name)
            if args.max_low is not None and len(low) > args.max_low:
                stopped = True
                break

    print()
    print(f"low detection ({len(low)}): {' '.join(low)}")
    print(f"failed ({len(failed)}): {' '.join(failed)}")
    if stopped:
        print(f"STOPPED: more than {args.max_low} low-detection clips")
    sys.exit(3 if stopped else 1 if failed else 0)


if __name__ == "__main__":
    main()
