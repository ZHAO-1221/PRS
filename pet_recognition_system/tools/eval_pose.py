"""Score RTMPose ONNX models on hand-labeled LabelMe images and compare them image by image.

    python tools/eval_pose.py P0=<data root>/models/rtmpose-m_ap10k/end2end.onnx [P1=checkpoints/p1_apt36k.onnx]

Images come from <data root>/评估 unless --eval-dir is given; the data root is --data-root or $PET_DATA_ROOT.
Every point present in a json is ground truth, a missing point is unlabeled. Each model predicts inside the
hand-drawn dog box, or with --boxes inside the detector boxes saved by tools/detect_boxes.py; an image whose
dog was not detected then scores 0. Scores are always normalized by the hand-drawn box, so runs with
different box sources are comparable. The first model is the baseline the others are compared with.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from pet_recognition.datasets.labelme_pose import KEYPOINTS, load_behavior_labels, load_labelme_pose
from pet_recognition.evaluation.keypoint_metrics import (average_precision, keypoint_similarity, paired_bootstrap,
                                                         pck)
from pet_recognition.models.rtmpose_onnx import RTMPoseOnnx

SAME = 0.01  # OKS changes smaller than this count as "same" in the paired comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", metavar="NAME=ONNX_PATH")
    parser.add_argument("--data-root", default=os.environ.get("PET_DATA_ROOT"))
    parser.add_argument("--eval-dir")
    parser.add_argument("--boxes", help="json from tools/detect_boxes.py; default: the hand-drawn boxes")
    parser.add_argument("--min-box-score", type=float, default=0.3)
    args = parser.parse_args()
    if not args.data_root:
        parser.error("set PET_DATA_ROOT or pass --data-root")
    eval_dir = Path(args.eval_dir or Path(args.data_root) / "评估")
    behavior_of = load_behavior_labels(args.data_root)
    detected = json.loads(Path(args.boxes).read_text(encoding="utf-8")) if args.boxes else None

    samples = []
    for path in sorted(eval_dir.glob("*.json")):
        rect, points = load_labelme_pose(path)
        gt = np.array([points.get(name, (np.nan, np.nan)) for name in KEYPOINTS])
        samples.append((path, rect, gt, behavior_of(path.stem)))
    behaviors = np.array([s[3] for s in samples])
    labeled = sum(int((~np.isnan(s[2][:, 0])).sum()) for s in samples)
    print(f"{len(samples)} images, {labeled} labeled keypoints, from {eval_dir}")
    print(f"pose input boxes: {args.boxes or 'hand-drawn'}")

    oks_by_model = {}
    for spec in args.models:
        name, _, onnx_path = spec.partition("=")
        model = RTMPoseOnnx(onnx_path)
        sim, dist, score = [], [], []
        for path, rect, gt, _ in samples:
            (xa, ya), (xb, yb) = rect
            if detected:
                dog = detected[path.stem]["dog"]
                rect = dog and dog[4] >= args.min_box_score and (dog[:2], dog[2:4])
            if rect:
                pred = model(Image.open(path.with_suffix(".jpg")), rect)
            else:  # no dog box: every labeled point is a miss
                pred = np.column_stack([np.full((len(KEYPOINTS), 2), np.inf), np.zeros(len(KEYPOINTS))])
            # box area stands in for the COCO segment area; PCK is normalized by the box's long side
            sim.append(keypoint_similarity(pred[:, :2], gt, abs(xb - xa) * abs(yb - ya)))
            dist.append(np.sqrt(((pred[:, :2] - gt) ** 2).sum(1)) / max(abs(xb - xa), abs(yb - ya)))
            score.append(pred[:, 2].mean())
        sim, dist = np.array(sim), np.array(dist)
        oks = oks_by_model[name] = np.nanmean(sim, axis=1)
        aps = average_precision(oks, np.array(score))

        print(f"\n== {name}  ({onnx_path})")
        print(f"OKS AP {aps.mean():.3f} | AP50 {aps[0]:.3f} | AP75 {aps[5]:.3f} | mean OKS {oks.mean():.3f}"
              f" | PCK@0.05 {pck(dist, 0.05):.3f} | PCK@0.1 {pck(dist, 0.1):.3f}")
        for b in sorted(set(behaviors)):
            pick = behaviors == b
            print(f"  {b:<16} n={pick.sum():<2} mean OKS {oks[pick].mean():.3f}  PCK@0.1 {pck(dist[pick], 0.1):.3f}")
        print("  PCK@0.1 per keypoint: " + "  ".join(f"{k} {pck(dist[:, i], 0.1):.2f}" for i, k in enumerate(KEYPOINTS)))

    base_name, *others = oks_by_model
    for name in others:
        diff = oks_by_model[name] - oks_by_model[base_name]
        mean, low, high = paired_bootstrap(diff)
        print(f"\n== {name} vs {base_name}: per-image OKS change")
        print(f"mean {mean:+.3f}, 95% CI [{low:+.3f}, {high:+.3f}] | better {(diff > SAME).sum()}, "
              f"same {(abs(diff) <= SAME).sum()}, worse {(diff < -SAME).sum()}")
        by_behavior = defaultdict(list)
        for b, d in zip(behaviors, diff):
            by_behavior[b].append(d)
        print("  by behavior: " + "  ".join(f"{b} {np.mean(v):+.3f}" for b, v in sorted(by_behavior.items())))
        for i in np.argsort(-abs(diff))[:5]:
            print(f"  {samples[i][0].stem:<24} {oks_by_model[base_name][i]:.3f} -> {oks_by_model[name][i]:.3f}")


if __name__ == "__main__":
    main()
