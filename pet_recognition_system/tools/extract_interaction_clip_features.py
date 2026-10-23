"""Cache frozen CLIP visual and prompt features for interaction clips."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoTokenizer, CLIPModel


MEAN = np.asarray((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
STD = np.asarray((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)
PROMPTS = (
    "a person and a dog are not interacting",
    "a person and a dog are resting separately",
    "a dog is near a person without physical contact",
    "a person ignores a nearby dog",
    "a person feeds a dog by hand",
    "a dog takes a treat from a person's hand",
    "a person gives food to a dog",
    "a dog's mouth touches food in a person's hand",
    "a person pets a dog",
    "a person strokes a dog's body",
    "a person hugs or holds a dog",
    "a person's hand stays in contact with a dog",
    "a person and a dog play together",
    "a person plays tug with a dog",
    "a person and a dog move energetically together",
    "a person uses a toy to play with a dog",
)


def preprocess(images):
    values = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = (image.astype(np.float32) / 255.0 - MEAN) / STD
        values.append(image.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(values))


def union_crop(frame, boxes, expansion=0.18):
    height, width = frame.shape[:2]
    valid = [box for box in boxes if box[2] > box[0] and box[3] > box[1]]
    if not valid:
        return frame
    x1 = min(float(box[0]) for box in valid)
    y1 = min(float(box[1]) for box in valid)
    x2 = max(float(box[2]) for box in valid)
    y2 = max(float(box[3]) for box in valid)
    pad_x, pad_y = (x2 - x1) * expansion, (y2 - y1) * expansion
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right, bottom = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
    return frame[top:bottom, left:right] if right > left and bottom > top else frame


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--skeleton-dir", required=True)
    parser.add_argument("--output-dir", default=str(project / "outputs/interaction_clip_features"))
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with Path(args.manifest).open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("valid", "1") == "1"]
    unique = {row["clip_id"]: row for row in rows}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", local_files_only=True
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "openai/clip-vit-base-patch32", local_files_only=True
    )
    tokens = tokenizer(list(PROMPTS), padding=True, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        text = model.get_text_features(**tokens)
        if not isinstance(text, torch.Tensor):
            text = text.pooler_output
        text = torch.nn.functional.normalize(text, dim=1)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    for number, clip_id in enumerate(sorted(unique), start=1):
        target = output / f"{clip_id}.npz"
        if target.exists():
            continue
        video = Path(args.video_dir) / f"{clip_id}.mp4"
        skeleton = Path(args.skeleton_dir) / f"interaction_{clip_id}.npz"
        if not video.exists() or not skeleton.exists():
            raise FileNotFoundError(f"missing video or skeleton for {clip_id}")
        with np.load(skeleton, allow_pickle=False) as archive:
            boxes = archive["boxes"]
        capture = cv2.VideoCapture(str(video))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        positions = np.linspace(0.0, 1.0, args.frames)
        indices = np.rint(positions * max(frame_count - 1, 0)).astype(int)
        box_indices = np.rint(positions * max(boxes.shape[1] - 1, 0)).astype(int)
        full_frames, crops, actual = [], [], []
        for index, box_index in zip(indices, box_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                full_frames.append(frame)
                crops.append(union_crop(frame, boxes[:, box_index]))
                actual.append(index)
        capture.release()
        if not full_frames:
            raise RuntimeError(f"could not decode {video}")
        pixels = preprocess(crops + full_frames).to(args.device)
        with torch.inference_mode():
            vision = model.vision_model(pixel_values=pixels).pooler_output
            projected = torch.nn.functional.normalize(model.visual_projection(vision), dim=1)
            prompt_scores = projected @ text.T
        count = len(full_frames)
        np.savez_compressed(
            target,
            crop_embeddings=vision[:count].cpu().numpy().astype(np.float32),
            full_embeddings=vision[count:].cpu().numpy().astype(np.float32),
            crop_prompt_scores=prompt_scores[:count].cpu().numpy().astype(np.float32),
            full_prompt_scores=prompt_scores[count:].cpu().numpy().astype(np.float32),
            frame_indices=np.asarray(actual, dtype=np.int32),
        )
        if number % 10 == 0 or number == len(unique):
            print(f"extracted {number}/{len(unique)}", flush=True)


if __name__ == "__main__":
    main()
