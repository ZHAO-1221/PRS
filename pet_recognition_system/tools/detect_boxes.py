"""Detect the dog in every image of a folder with RTMDet and save the boxes for tools/eval_pose.py --boxes.

    python tools/detect_boxes.py [--image-dir DIR] [--out outputs/eval_boxes_rtmdet.json]

Defaults to the images in <data root>/评估, which is only read. The output maps each image stem to
{"dog": [x1, y1, x2, y2, score, COCO class] or null, "person": the same}: the best box of each at any
score. The dog box may come from another animal class (see pet_recognition/models/rtmdet.py). Where an
image has a LabelMe json, the IoU between the detected and the hand-drawn dog box is reported.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from pet_recognition.datasets.labelme_pose import load_labelme_pose
from pet_recognition.models.rtmdet import RTMDet

MIN_SCORE = 0.3  # same default as tools/eval_pose.py --min-box-score


def iou(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    overlap = max(width, 0) * max(height, 0)
    return overlap / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap)


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=os.environ.get("PET_DATA_ROOT"))
    parser.add_argument("--image-dir")
    parser.add_argument("--out", default=str(project / "outputs" / "eval_boxes_rtmdet.json"))
    args = parser.parse_args()
    if not args.data_root:
        parser.error("set PET_DATA_ROOT or pass --data-root")
    root = Path(args.data_root)
    image_dir = Path(args.image_dir or root / "评估")
    weights = root / "models" / "rtmdet-m_coco" / "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
    detector = RTMDet(weights)

    boxes, ious = {}, []
    for path in sorted(image_dir.glob("*.jpg")):
        found = detector(str(path))
        dog, person = found["dog"], found["person"]
        boxes[path.stem] = {name: box and [*(round(v, 3) for v in box[:5]), box[5]] for name, box in found.items()}
        note = ""
        if path.with_suffix(".json").exists() and dog:
            (xa, ya), (xb, yb) = load_labelme_pose(path.with_suffix(".json"))[0]
            overlap = iou(dog, (min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)))
            if dog[4] >= MIN_SCORE:
                ious.append(overlap)
            note = f"IoU with hand box {overlap:.2f}"
        score = dog[4] if dog else 0
        flag = "" if score >= MIN_SCORE else "  <-- no dog box above the score threshold"
        print(f"{path.stem:<24} dog {score:.2f} (as {dog[5] if dog else None})  person {person[4] if person else 0:.2f}  {note}{flag}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(boxes, indent=1), encoding="utf-8")
    detected = sum(bool(b["dog"]) and b["dog"][4] >= MIN_SCORE for b in boxes.values())
    print(f"\ndog detected (score >= {MIN_SCORE}) in {detected} of {len(boxes)} images; wrote {args.out}")
    if ious:
        print(f"IoU with the hand-drawn box: median {np.median(ious):.2f}, min {min(ious):.2f}, "
              f"below 0.5 in {sum(v < 0.5 for v in ious)} images")


if __name__ == "__main__":
    main()
