"""Frozen CLIP encoder used by the released interaction fusion model."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


MEAN = np.asarray((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
STD = np.asarray((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)


def _union_crop(frame: np.ndarray, boxes: np.ndarray, expansion: float = 0.18) -> np.ndarray:
    height, width = frame.shape[:2]
    valid = [box for box in boxes if box[2] > box[0] and box[3] > box[1]]
    if not valid:
        return frame
    x1, y1 = min(float(box[0]) for box in valid), min(float(box[1]) for box in valid)
    x2, y2 = max(float(box[2]) for box in valid), max(float(box[3]) for box in valid)
    pad_x, pad_y = (x2 - x1) * expansion, (y2 - y1) * expansion
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right, bottom = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
    return frame[top:bottom, left:right] if right > left and bottom > top else frame


def _preprocess(images: list[np.ndarray]) -> torch.Tensor:
    values = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = (image.astype(np.float32) / 255.0 - MEAN) / STD
        values.append(image.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(values))


class InteractionVisualEncoder:
    """Return the 4,608-D frozen feature used to train the RGB branch."""

    def __init__(self, device: str = "cpu", model_name: str = "openai/clip-vit-base-patch32"):
        try:
            from transformers import CLIPModel
        except ImportError as error:
            raise RuntimeError(
                "multimodal interaction inference requires the transformers package"
            ) from error
        self.device = device
        self.model = CLIPModel.from_pretrained(
            model_name,
            local_files_only=Path(model_name).exists(),
        ).to(device).eval()

    def encode(self, frames: list[np.ndarray], boxes: np.ndarray, samples: int = 8) -> np.ndarray:
        if not frames:
            raise ValueError("cannot extract visual features from an empty window")
        usable = min(len(frames), boxes.shape[1])
        indices = np.linspace(0, max(usable - 1, 0), samples).round().astype(int)
        full = [frames[index] for index in indices]
        crops = [_union_crop(frames[index], boxes[:, index]) for index in indices]
        pixels = _preprocess(crops + full).to(self.device)
        with torch.inference_mode():
            embeddings = self.model.vision_model(pixel_values=pixels).pooler_output
        embeddings = embeddings.cpu().numpy().astype(np.float32)
        count = len(indices)
        values = []
        for part in (embeddings[:count], embeddings[count:]):
            values.extend((part.mean(0), part.std(0), np.abs(part[-1] - part[0])))
        return np.concatenate(values).astype(np.float32)
