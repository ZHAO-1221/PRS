# Pet Recognition System

Skeleton-based dog behaviour and human-dog interaction recognition project.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the repository structure and directory responsibilities.

## Setup

```bash
conda create -n PRS-project python=3.10 -y
conda activate PRS-project
pip install -r requirements.txt          # keypoint evaluation and ONNX inference
pip install -e .
```

Fine-tuning RTMPose also needs the OpenMMLab stack. This is the CPU build: OpenMMLab publishes no Windows mmcv
wheel for the CUDA 12.8 PyTorch an RTX 50 card needs. mmpose is installed with `--no-deps` because its chumpy
dependency does not build with current pip and is not needed here.

```bash
pip install "setuptools<81"
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
pip install mmengine==0.10.7 onnx==1.23.2
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html
pip install mmdet==3.2.0 json_tricks==3.17.3 munkres==2.0.0 xtcocotools==1.14.3 albumentations==1.3.1
pip install mmpose==1.3.2 --no-deps
# albumentations pulls in NumPy 2 and a headless OpenCV that break torch 2.1; put the pinned ones back
pip uninstall -y opencv-python-headless opencv-python
pip install numpy==1.26.4 opencv-python==4.11.0.86
```

Data and model weights are not in the repository. Every script finds them through the `PET_DATA_ROOT`
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

## Skeleton sequences for model two

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

A stream is all zeros (confidence 0) in frames where its target was not detected, e.g. the person stream of a
dog-only clip. Coordinates are not normalized and can fall outside the image for body parts out of frame.
When several people or dogs are in view, the first frame takes the most confident box; every later frame takes
the box overlapping the previous one most (a box scoring only 0.1-0.3, as lying dogs often do, counts when it
overlaps the previous box by IoU >= 0.5) and counts the target as not detected when no box qualifies. After 3
missed frames in a row the track is dropped and the most confident box is taken again. `--video` also writes `<clip_id>_skeleton.mp4` for checking by eye; use it on a review sample (about 15 clips
spread evenly over the behavior and interaction classes), not on the whole batch.

Tests: `python tests/test_keypoint_metrics.py`, `python tests/test_extract_skeletons.py`, `python tests/test_rtmdet_choose.py`.
