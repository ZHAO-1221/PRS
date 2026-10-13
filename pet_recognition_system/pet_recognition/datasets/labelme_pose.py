"""LabelMe dog-pose labels: one `dog` rectangle plus named AP-10K keypoints per image."""
import csv
import json
from pathlib import Path

# AP-10K order, spelled as the points are named in LabelMe
KEYPOINTS = ["L_Eye", "R_Eye", "Nose", "Neck", "Root_of_tail", "L_Shoulder", "L_Elbow", "L_F_Paw",
             "R_Shoulder", "R_Elbow", "R_F_Paw", "L_Hip", "L_Knee", "L_B_Paw", "R_Hip", "R_Knee", "R_B_Paw"]


def load_labelme_pose(json_path):
    """Return (rectangle corner points, {keypoint name: (x, y)}); a point missing from the file is unlabeled."""
    shapes = json.loads(Path(json_path).read_text(encoding="utf-8"))["shapes"]
    rects = [s["points"] for s in shapes if s["shape_type"] == "rectangle"]
    points = [(s["label"], tuple(s["points"][0])) for s in shapes if s["shape_type"] == "point"]
    assert len(rects) == 1, f"{json_path}: expected exactly one dog box, found {len(rects)}"
    assert len(dict(points)) == len(points), f"{json_path}: a keypoint is labeled twice"
    assert set(dict(points)) <= set(KEYPOINTS), f"{json_path}: unknown labels {set(dict(points)) - set(KEYPOINTS)}"
    return rects[0], dict(points)


def load_behavior_labels(data_root):
    """Return a function mapping an image stem ({clip_id}_f{N}) to the behavior label of its clip."""
    data_root = Path(data_root)
    clips = {}
    for split in ("train", "test"):
        with open(data_root / "raw" / f"indoor {split} raw data" / "dog_behavior_clips.csv", encoding="utf-8-sig") as f:
            clips.update((row["clip_id"], row["behavior_label"]) for row in csv.DictReader(f))
    # two-dog frames were cropped into single-dog images; crops.csv maps each crop back to its source frame
    with open(data_root / "images" / "crops.csv", encoding="utf-8-sig") as f:
        source = {Path(row["file"]).stem: Path(row["source_file"]).stem for row in csv.DictReader(f)}
    return lambda stem: clips[source.get(stem, stem).rsplit("_f", 1)[0]]
