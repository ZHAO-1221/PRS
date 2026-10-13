"""Clean the APT-36K dog annotations and split them into train / val by source video.

    python tools/prepare_apt36k_split.py

Reads <data root>/apt36k_annotations.json (dog-only COCO, images under <data root>/APT-dog/<clip>/) and writes
apt36k_dog_train.json and apt36k_dog_val.json next to it, with image paths relative to the data root.
The data root is --data-root or $PET_DATA_ROOT.
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path, PureWindowsPath

VAL_SHARE = 0.15


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=os.environ.get("PET_DATA_ROOT"))
    args = parser.parse_args()
    if not args.data_root:
        parser.error("set PET_DATA_ROOT or pass --data-root")
    root = Path(args.data_root)
    coco = json.loads((root / "apt36k_annotations.json").read_text(encoding="utf-8"))

    annotations = []
    for ann in coco["annotations"]:
        visible = sum(v > 0 for v in ann["keypoints"][2::3])
        if visible:  # boxes without any keypoint cannot train or score a pose model
            annotations.append({**ann, "num_keypoints": visible, "iscrowd": 0})

    images, by_video = {}, defaultdict(list)
    for image in coco["images"]:
        clip, frame = PureWindowsPath(image["file_name"]).parts[-2:]  # the json holds absolute Windows paths
        images[image["id"]] = {**image, "file_name": f"APT-dog/{clip}/{frame}"}
        # clips cut from one video (v18c1, v18c2, ...) must stay on the same side of the split
        by_video[re.sub(r"(?i)^(v\d+)c\d+$", r"\1", clip).lower()].append(image["id"])

    videos = sorted(by_video)
    random.Random(0).shuffle(videos)
    val_ids = set()
    for video in videos:
        if len(val_ids) >= VAL_SHARE * len(images):
            break
        val_ids.update(by_video[video])

    for split, keep in (("train", set(images) - val_ids), ("val", val_ids)):
        anns = [a for a in annotations if a["image_id"] in keep]
        used = sorted({a["image_id"] for a in anns})
        out = {**coco, "images": [images[i] for i in used], "annotations": anns}
        (root / f"apt36k_dog_{split}.json").write_text(json.dumps(out), encoding="utf-8")
        split_videos = sorted(v for v in by_video if set(by_video[v]) <= keep)
        print(f"{split}: {len(split_videos)} videos, {len(used)} images, {len(anns)} dogs  {split_videos}")
    assert all((root / images[i]["file_name"]).exists() for i in images), "an image file is missing under APT-dog/"


if __name__ == "__main__":
    main()
