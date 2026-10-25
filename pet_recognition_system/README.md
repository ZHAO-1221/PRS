# Pet Recognition System

Skeleton-based dog behaviour and human-dog interaction recognition project.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the repository structure and directory responsibilities.
The selected recognition weights and reports are documented in
[checkpoints/best_models](checkpoints/best_models/README.md), and exact
retraining commands are in [MODEL_TRAINING.md](MODEL_TRAINING.md).

## Setup

```bash
conda create -n PRS-project-311 python=3.11 -y
conda activate PRS-project-311
pip install -r requirements.txt          # keypoint evaluation and ONNX inference
pip install -e .
```

Python 3.11 is required: the released scikit-learn models in `checkpoints/best_models/` were saved with
scikit-learn 1.9.1 (pinned in `requirements-training.txt`), which needs Python 3.11+, and pickled models only load
reliably with the version that saved them. Re-export the `.joblib` files and update the pin together if either
changes.

Fine-tuning RTMPose also needs the OpenMMLab stack. This is the CPU build: OpenMMLab publishes no Windows mmcv
wheel for the CUDA 12.8 PyTorch an RTX 50 card needs. mmpose is installed with `--no-deps` because its chumpy
dependency does not build with current pip and is not needed here.

```bash
pip install "setuptools<81"
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
pip install mmengine==0.10.7 onnx==1.23.2 ml-dtypes==0.5.4   # newer ml-dtypes wants NumPy 2
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html
pip install mmdet==3.2.0 json_tricks==3.17.3 munkres==2.0.0 xtcocotools==1.14.3 albumentations==1.3.1
pip install mmpose==1.3.2 --no-deps
# albumentations pulls in NumPy 2 and a headless OpenCV that break torch 2.1; put the pinned ones back
pip uninstall -y opencv-python-headless opencv-python
pip install numpy==1.26.4 opencv-python==4.11.0.86
```

Source data and pose-model weights are not in the repository. Every script finds them through the `PET_DATA_ROOT`
environment variable (or `--data-root`), which points at the local data folder:

```text
<PET_DATA_ROOT>/
├── 评估/                         hand-labeled LabelMe images used to compare pose models
├── raw/dog_behavior/             behavior clips (<clip_id>.mp4) and dog_behavior_clips.csv
├── APT-dog/                      APT-36K dog frames
├── apt36k_annotations.json       APT-36K dog annotations (COCO)
├── models/rtmpose-m_ap10k/       official RTMPose-m AP-10K weights (.pth and end2end.onnx)
└── models/rtmdet-m_coco/         official RTMDet-m COCO weights (.pth)
```

## Dog pose model (model one)

Score a pose model on the hand-labeled evaluation images; give several models to compare them with the first:

```bash
python tools/eval_pose.py P0=$PET_DATA_ROOT/models/rtmpose-m_ap10k/end2end.onnx
python tools/eval_pose.py P0=$PET_DATA_ROOT/models/rtmpose-m_ap10k/end2end.onnx P1=checkpoints/p1_apt36k.onnx
```

By default the pose models predict inside the hand-drawn dog boxes. To score the real pipeline, detect the dogs
with RTMDet first (needs the fine-tuning stack) and pass the saved boxes; `--min-box-score 0` keeps the best
dog box of every image whatever its score:

```bash
python tools/detect_boxes.py                                # writes outputs/eval_boxes_rtmdet.json
python tools/eval_pose.py P0=$PET_DATA_ROOT/models/rtmpose-m_ap10k/end2end.onnx --boxes outputs/eval_boxes_rtmdet.json
```

Fine-tune the official AP-10K model on the APT-36K dog subset (P1):

```bash
python tools/prepare_apt36k_split.py                                              # train / val split by video
python tools/train_pose.py configs/rtmpose/rtmpose-m_apt36k-dog_finetune.py --test   # starting weights on APT val
python tools/train_pose.py configs/rtmpose/rtmpose-m_apt36k-dog_finetune.py          # train, best epoch kept in work_dirs/
python tools/export_rtmpose_onnx.py configs/rtmpose/rtmpose-m_apt36k-dog_finetune.py \
    work_dirs/rtmpose-m_apt36k-dog_finetune/best_coco_AP_epoch_N.pth checkpoints/p1_apt36k.onnx
```

## Skeleton sequences for recognition

```bash
python tools/extract_skeletons.py CLIP.mp4 [CLIP.mp4 ...] --out-dir outputs/skeletons [--video]
```

Each clip becomes `<clip_id>.npz`, resampled to 15 fps:

| key | shape | content |
|---|---|---|
| `skeletons` | (2, T, 17, 3) float32 | stream 0 person (COCO keypoint order), stream 1 dog (AP-10K order); x, y in pixels, confidence in (0, 1] |
| `boxes` | (2, T, 5) float32 | person / dog box x1, y1, x2, y2, detector score |
| `fps` | scalar | 15 |
| `image_size` | (2,) | width, height in pixels |

A stream is all zeros (confidence 0) in frames where its target was not detected. Clips in a `dog_behavior` folder get only the dog
stream (their person stream is all zeros throughout); clips in an `interaction` folder get both. Coordinates are not normalized and can fall outside the image for body parts out of frame.
When several people or dogs are in view, the first frame takes the most confident box; every later frame takes
the box overlapping the previous one most (a box scoring only 0.1-0.3, as lying dogs often do, counts when it
overlaps the previous box by IoU >= 0.3) and counts the target as not detected when no box qualifies. Through
missed frames the last box seen is kept, so the same rule picks the target up again where it was last seen (a dog
hidden behind or jumping on its owner comes back at 0.1-0.3); after 3 missed frames in a row a box scoring 0.3
anywhere may also take the track over. A target that never scores 0.3 in a clip (a far or lying dog
often stays at 0.1-0.3) starts instead from the highest-scoring box of the whole clip. Either way the track is also
followed backwards from its start frame to the beginning of the clip. `--low-score 0` drops the 0.1 floor for
weak boxes, for a clip where the detector finds the dog but never scores it well. It was tried on
`d025_v24_c002` (night infrared, a curled-up dog scored 0.04-0.3 as a cat): the skeletons were unusable, so that
clip is now valid=0 and has no `.npz`. `--video` also writes `<clip_id>_skeleton.mp4` for checking by eye; use it on a review sample (about 15 clips
spread evenly over the behavior and interaction classes), not on the whole batch.

## Selected recognition models

Only the best-performing configuration for each active recognition task is
published. The fold files are components of one ensemble per task, not model
history. Superseded ST-GCN, LSTM, Transformer, RGB-fusion, hierarchical and
ablation checkpoints are excluded.

- Dog behaviour: motion-aware TCN with geometry fusion, five classes.
- Human-dog interaction: relation-feature ensemble fused with a frozen CLIP
  visual branch, four classes.

Install `requirements-training.txt`, then follow [MODEL_TRAINING.md](MODEL_TRAINING.md)
to retrain or evaluate the published ensembles.

## End-to-end video prediction

Predict both dog behaviour and human-dog interaction from one video:

```bash
python tools/predict_video.py input.mp4 --task both --data-root <PET_DATA_ROOT>
```

Other Python applications, including an HTTP service, can call the same public
entry point directly:

```python
from tools.predict_video import predict_video

result = predict_video(
    input_path="uploads/example.mp4",
    output_dir="outputs/example",
    data_root="path/to/data",
    task="both",
)
```

The data root must contain the three pose-pipeline weights documented above:

```text
<PET_DATA_ROOT>/models/
├── rtmdet-m_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth
├── rtmpose-m_coco_human/end2end.onnx
└── rtmpose-m_ap10k/end2end.onnx
```

Interaction prediction uses the frozen official
[`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)
model. If `<PET_DATA_ROOT>/models/clip-vit-base-patch32/` contains a complete
local snapshot, it is used first. Otherwise Transformers downloads the official
model on the first run and reuses its local Hugging Face cache afterwards. A
local snapshot is therefore only required for offline inference.

Videos up to eight seconds are classified as one clip. Longer videos are read
lazily and divided into four-second windows, so a long recording is not loaded
fully into memory. The command writes `result.json` with window probabilities
and merged timelines, plus `timeline.csv` with one row per time segment. Use
`--export-clips` to render each merged segment with FFmpeg. Other useful modes:

```bash
python tools/predict_video.py input.mp4 --task dog_behavior --data-root <PET_DATA_ROOT>
python tools/predict_video.py input.mp4 --task interaction --data-root <PET_DATA_ROOT>
python tools/predict_video.py skeleton_window.npz --task both
```

The NPZ form bypasses pose extraction and is useful for testing an existing
`skeletons`, `boxes`, `fps`, `image_size` record against the released models.

Tests: `python -m pytest -q`.

## Mobile app backend

`app/server.py` is a FastAPI service around `predict_video()` for the mobile frontend in `../frontend`.
Install its extra dependencies, then run it from this folder:

```bash
pip install -r requirements-app.txt
PET_DATA_ROOT=<data-root> uvicorn app.server:app --port 8000
```

| Endpoint | Purpose |
|---|---|
| `POST /api/clips` | multipart `video`, `recorded_at` (`YYYY-MM-DDTHH:MM`), `camera`; returns `{job_id, clip_id}` and analyses in the background |
| `GET /api/jobs/{job_id}` | `status` (`queued`/`running`/`done`/`failed`), `step` (finished steps, 0-4), `progress`, `error`, `stages` (per stage `name`, `status` `pending`/`running`/`done`/`failed`, `done`/`total`) |
| `GET /api/clips?from=&to=` | analysed clips in a date range, results without per-window data |
| `GET /api/clips/{clip_id}` | one clip with its full `result.json` |
| `GET /api/clips/{clip_id}/cover` | JPEG cover taken about one second into the video |
| `GET /api/models/metrics` | test accuracy, macro-F1 and test clip count from `outputs/best_models/` |

Uploads are limited to 500 MB of mp4/mov/avi/mkv/m4v and are stored under `outputs/app_store/clips/<clip_id>/`
(override with `PET_APP_STORE`). One clip is analysed at a time; `PET_DEVICE` picks the inference device
(default `cpu`). Tests: `python -m pytest tests/test_app_api.py`.
