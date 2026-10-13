"""RTMDet-m (COCO weights) through MMDetection, reduced to the two things this project tracks: a person and a dog."""
from mmdet.apis import inference_detector, init_detector
from mmengine.hub import get_config

CONFIG = "mmdet::rtmdet/rtmdet_m_8xb32-300e_coco.py"
# The COCO model often scores an indoor dog higher as another four-legged animal (mostly "cat").
# shortcut: assumes the dog is the only animal in view; a real cat would be returned as the dog.
# Fine-tune a two-class detector if that stops holding.
DOG_LIKE = {"dog", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}


class RTMDet:
    def __init__(self, checkpoint, device="cpu"):
        self.model = init_detector(get_config(CONFIG), str(checkpoint), device=device)
        self.classes = self.model.dataset_meta["classes"]

    def __call__(self, image):
        """Best person box and best dog box, each (x1, y1, x2, y2, score, COCO class) or None.

        image: path or BGR array. Boxes come back at any score; the caller applies its own threshold.
        """
        found = inference_detector(self.model, image).pred_instances
        best = {"person": None, "dog": None}
        for box, score, label in zip(found.bboxes.tolist(), found.scores.tolist(), found.labels.tolist()):
            name = self.classes[label]
            target = "person" if name == "person" else "dog" if name in DOG_LIKE else None
            if target and (best[target] is None or score > best[target][4]):
                best[target] = (*box, score, name)
        return best
