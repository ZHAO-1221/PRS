"""Show where the pipeline's person boxes come from, to find out why it reports people that are not there.

    python tools/diagnose_person_detections.py VIDEO [VIDEO ...] [--out-dir outputs/person_fp_diagnosis]

Runs the same windows, RTMDet candidates and follow() tracking as tools/predict_video.py (task both) and,
for every frame where a person box is chosen, records why it was taken and what it looks like:
<out-dir>/<video>/person_frames.csv  one row per frame with a person box: window, time, score, COCO class,
    path (confident = score >= MIN_BOX_SCORE; weak = a 0.1-0.3 box continuing a track; weak_seed = the window
    had no person box at the start score in STREAM_SCORES, so its track started from the best weak box), IoU with the dog box,
    share of the person box inside the dog box, box area / image area, mean person keypoint confidence;
<out-dir>/<video>/person_crops.jpg   up to 24 crops of chosen person boxes, labelled, to mark true / false by eye;
<out-dir>/summary.csv                 one row per video.
Weights come from <data root>/models ($PET_DATA_ROOT). Nothing in the pipeline is changed.
"""
import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pet_recognition.inference.skeleton import MIN_BOX_SCORE, STREAM_SCORES, follow
from pet_recognition.inference.video_pipeline import iter_video_windows
from pet_recognition.models.rtmdet import RTMDet, iou
from pet_recognition.models.rtmpose_onnx import RTMPoseOnnx

FIELDS = ("window", "time_sec", "score", "coco_class", "path", "dog_iou", "inside_dog", "area_share",
          "keypoint_conf", "window_person_rate")
CROPS, CROP_HEIGHT, COLUMNS = 24, 180, 6


def inside(a, b):
    """Share of box a covered by box b."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return max(width, 0) * max(height, 0) / max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)


def crop_sheet(crops):
    """Each crop fitted into a square tile with its label above, COLUMNS tiles per row."""
    tiles = []
    for frame, box, label in crops:
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        crop = frame[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)]
        scale = CROP_HEIGHT / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
        tile = np.zeros((CROP_HEIGHT + 22, CROP_HEIGHT, 3), np.uint8)
        tile[22:22 + crop.shape[0], :crop.shape[1]] = crop
        cv2.putText(tile, label, (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        tiles.append(tile)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % COLUMNS)
    return np.vstack([np.hstack(tiles[i:i + COLUMNS]) for i in range(0, len(tiles), COLUMNS)])


def diagnose(video, detector, person_pose, out_dir):
    rows, crops = [], []
    for w, window in enumerate(iter_video_windows(video)):
        frames = window["frames"]
        candidates = [detector.candidates(frame) for frame in frames]
        person = follow([c["person"] for c in candidates], *STREAM_SCORES["person"])
        dog = follow([c["dog"] for c in candidates], *STREAM_SCORES["dog"])
        weak_seed = not any(b[4] >= STREAM_SCORES["person"][0] for c in candidates for b in c["person"])
        rate = float(np.mean([b is not None for b in person]))
        for t, (frame, box) in enumerate(zip(frames, person)):
            if box is None:
                continue
            image = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))
            keypoints = person_pose(image, (box[:2], box[2:4]))
            height, width = frame.shape[:2]
            row = {"window": w, "time_sec": round(window["start_sec"] + t / 15, 2), "score": round(box[4], 3),
                   "coco_class": box[5],
                   "path": "weak_seed" if weak_seed else "confident" if box[4] >= MIN_BOX_SCORE else "weak",
                   "dog_iou": round(iou(box, dog[t]), 3) if dog[t] else "",
                   "inside_dog": round(inside(box, dog[t]), 3) if dog[t] else "",
                   "area_share": round((box[2] - box[0]) * (box[3] - box[1]) / (width * height), 4),
                   "keypoint_conf": round(float(np.clip(keypoints[:, 2], 0, 1).mean()), 3),
                   "window_person_rate": round(rate, 2)}
            rows.append(row)
            crops.append((frame, box, f"{row['time_sec']}s {row['score']:.2f} {row['path']}"))
    target = out_dir / video.stem
    target.mkdir(parents=True, exist_ok=True)
    with (target / "person_frames.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if crops:
        picked = [crops[i] for i in np.linspace(0, len(crops) - 1, min(CROPS, len(crops))).astype(int)]
        ok, data = cv2.imencode(".jpg", crop_sheet(picked))
        data.tofile(str(target / "person_crops.jpg"))  # tofile: imwrite cannot write non-ASCII paths on Windows
    return rows


def median(values):
    return round(float(np.median(values)), 3) if values else ""


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--data-root", default=os.environ.get("PET_DATA_ROOT"))
    parser.add_argument("--out-dir", type=Path, default=project / "outputs" / "person_fp_diagnosis")
    args = parser.parse_args()
    if not args.data_root:
        parser.error("set PET_DATA_ROOT or pass --data-root")
    models = Path(args.data_root) / "models"
    detector = RTMDet(models / "rtmdet-m_coco" / "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth")
    person_pose = RTMPoseOnnx(models / "rtmpose-m_coco_human" / "end2end.onnx")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for video in args.videos:
        rows = diagnose(video, detector, person_pose, args.out_dir)
        paths = [r["path"] for r in rows]
        line = {"video": video.stem, "folder": video.parent.name, "person_frames": len(rows),
                **{p: paths.count(p) for p in ("confident", "weak", "weak_seed")},
                "median_score": median([r["score"] for r in rows]),
                "median_keypoint_conf": median([r["keypoint_conf"] for r in rows]),
                "frames_inside_dog_over_0.5": sum(r["inside_dog"] != "" and r["inside_dog"] > 0.5 for r in rows),
                "classes": " ".join(sorted({r["coco_class"] for r in rows}))}
        summary.append(line)
        print(line, flush=True)
    with (args.out_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
