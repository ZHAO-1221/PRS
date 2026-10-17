"""RTMDet-m (COCO weights) through MMDetection, reduced to the two things this project tracks: a person and a dog."""
from mmdet.apis import inference_detector, init_detector
from mmengine.hub import get_config

CONFIG = "mmdet::rtmdet/rtmdet_m_8xb32-300e_coco.py"
# The COCO model often scores an indoor dog higher as another four-legged animal (mostly "cat").
# shortcut: assumes the dog is the only animal in view; a real cat would be returned as the dog.
# Fine-tune a two-class detector if that stops holding.
DOG_LIKE = {"dog", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
FOLLOW_MIN_SCORE = 0.3  # boxes below this are not real detections; same as the box threshold of the pipeline
# a lying dog often scores only 0.1-0.3 while staying where it was, so a weak box may continue a track if it
# overlaps the last box this much
FOLLOW_LOW_SCORE, FOLLOW_LOW_MIN_IOU = 0.1, 0.3


def iou(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    overlap = max(width, 0) * max(height, 0)
    return overlap / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap)


def choose(boxes, last=None):
    """First appearance (no last box): the highest-scoring candidate. After that: of the confident candidates
    overlapping the last box, or the weak ones overlapping it a lot, the one overlapping it most; None when
    there is none, so another person or dog never takes over."""
    if last is None:
        return max(boxes, key=lambda b: b[4], default=None)
    near = [b for b in boxes if (b[4] >= FOLLOW_MIN_SCORE and iou(b, last) > 0)
            or (b[4] >= FOLLOW_LOW_SCORE and iou(b, last) >= FOLLOW_LOW_MIN_IOU)]
    return max(near, key=lambda b: iou(b, last), default=None)


class RTMDet:
    def __init__(self, checkpoint, device="cpu"):
        self.model = init_detector(get_config(CONFIG), str(checkpoint), device=device)
        self.classes = self.model.dataset_meta["classes"]

    def candidates(self, image):
        """Every person box and every dog box found: {"person": [...], "dog": [...]}, each box
        (x1, y1, x2, y2, score, COCO class) at any score. image: path or BGR array."""
        found = inference_detector(self.model, image).pred_instances
        candidates = {"person": [], "dog": []}
        for box, score, label in zip(found.bboxes.tolist(), found.scores.tolist(), found.labels.tolist()):
            name = self.classes[label]
            target = "person" if name == "person" else "dog" if name in DOG_LIKE else None
            if target:
                candidates[target].append((*box, score, name))
        return candidates

    def __call__(self, image, previous=None):
        """Person box and dog box, each (x1, y1, x2, y2, score, COCO class) or None.

        image: path or BGR array. Boxes come back at any score; the caller applies its own threshold.
        previous: {"person": box, "dog": box} kept from earlier frames of the same video. A target without a
        previous box takes the highest-scoring candidate; one with a previous box only takes the candidate
        overlapping it most, so a second person or dog in view never takes over the track.
        """
        return {target: choose(boxes, (previous or {}).get(target)) for target, boxes in self.candidates(image).items()}
