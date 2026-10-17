"""Frames -> person and dog skeletons: RTMDet finds both, RTMPose estimates 17 keypoints for each."""
import numpy as np
from PIL import Image

from pet_recognition.models import rtmdet
from pet_recognition.models.rtmdet import choose

STREAMS = ("person", "dog")  # stream order in the output arrays
MIN_BOX_SCORE = 0.3  # a new target needs this score; a followed one is checked by RTMDet's choose() instead
REACQUIRE_AFTER = 3  # missed frames after which a confident box anywhere may take over the track


def track(per_frame, seed=None):
    """Follow one target through a clip. per_frame: its candidate boxes in each frame.

    Without a box to follow, a frame takes its highest-scoring candidate if it scores MIN_BOX_SCORE; while
    following, RTMDet's choose() picks the candidate overlapping the last box seen. The last box is kept
    through misses, so a weak box where the target was last seen still picks the track up again (a dog hidden
    behind or jumping on its owner comes back at 0.1-0.3); after REACQUIRE_AFTER missed frames a confident box
    elsewhere may also start it again. seed: a box to follow from the first frame on. Returns a box or None
    per frame.
    """
    chosen, last, missed = [], seed, 0
    for boxes in per_frame:
        box = choose(boxes, last) if last is not None else None
        if box is None and (last is None or missed >= REACQUIRE_AFTER):  # lost, e.g. the box jumped
            best = max(boxes, key=lambda b: b[4], default=None)
            box = best if best is not None and best[4] >= MIN_BOX_SCORE else None
        chosen.append(box)
        if box is None:
            missed += 1
        else:
            last, missed = box, 0
    return chosen


def follow(per_frame):
    """Pick one target's box in every frame of a clip: start from the first frame with a box scoring
    MIN_BOX_SCORE (its best box), or, if no frame has one, from the clip's highest-scoring box (a far or lying
    dog often stays at 0.1-0.3); then track() from there forwards and also backwards to the clip's start, so the
    frames before the target first scores well are not lost."""
    confident = [t for t, boxes in enumerate(per_frame) if any(b[4] >= MIN_BOX_SCORE for b in boxes)]
    if confident:
        start = confident[0]
        seed = max(per_frame[start], key=lambda b: b[4])
    else:
        best = max(((b, t) for t, boxes in enumerate(per_frame) for b in boxes), key=lambda x: x[0][4], default=None)
        if best is None or best[0][4] < rtmdet.FOLLOW_LOW_SCORE:
            return [None] * len(per_frame)
        seed, start = best
    before = track(per_frame[start::-1], seed)[::-1]  # from the start frame back to the first one
    return before[:-1] + track(per_frame[start:], seed)


def extract_skeletons(frames, detector, pose_models, streams=STREAMS):
    """Run the detector and the pose models on BGR frames of one clip.

    pose_models: {"person": RTMPoseOnnx (COCO 17 points), "dog": RTMPoseOnnx (AP-10K 17 points)}.
    streams: the targets to extract, e.g. ("dog",) for dog-only clips; the other stream stays all zeros.
    Returns skeletons (2, T, 17, 3) float32 of x, y in pixels and a confidence in (0, 1], and boxes (2, T, 5)
    of x1, y1, x2, y2, score; a stream whose target was not detected in a frame stays all zeros there.
    """
    candidates = [detector.candidates(frame) for frame in frames]  # the whole clip first: follow() looks ahead
    tracks = {stream: follow([c[stream] for c in candidates]) for stream in streams}
    skeletons = np.zeros((2, len(frames), 17, 3), np.float32)
    boxes = np.zeros((2, len(frames), 5), np.float32)
    for t, frame in enumerate(frames):
        image = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))
        for i, stream in enumerate(STREAMS):
            box = tracks[stream][t] if stream in tracks else None
            if box is None:
                continue
            skeletons[i, t] = pose_models[stream](image, (box[:2], box[2:4]))
            # RTMPose's SimCC peak is not a probability: it can exceed 1 or dip just below 0. Keep it in
            # (0, 1] so that 0 still means "not detected"
            skeletons[i, t, :, 2] = np.clip(skeletons[i, t, :, 2], 0.001, 1)
            boxes[i, t] = box[:5]
    return skeletons, boxes
