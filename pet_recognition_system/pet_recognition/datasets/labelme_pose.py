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
    with open(Path(data_root) / "raw" / "dog_behavior" / "dog_behavior_clips.csv", encoding="utf-8-sig") as f:
        clips = {row["clip_id"]: row["behavior_label"] for row in csv.DictReader(f)}
    same_video_and_clip = {}
    for clip_id in clips:
        same_video_and_clip.setdefault(clip_id.split("_", 1)[1], []).append(clip_id)

    def label(stem):
        dog, video, number = stem.rsplit("_f", 1)[0].split("_")
        if video.startswith("t"):  # test videos t01-t05 are v27-v31 in the clip list
            video = f"v{int(video[1:]) + 26:02d}"
        clip_id = f"{dog}_{video}_{number}"
        if clip_id not in clips:
            # a second dog cropped out of a two-dog frame got a new dog_id but kept the video and clip numbers
            (clip_id,) = same_video_and_clip[f"{video}_{number}"]
        return clips[clip_id]

    return label
