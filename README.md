# Skeleton-Based Dog Behaviour and Human-Dog Interaction Recognition

**Group 28 · NUS PRS Practice Module**

## Project Team Members

| **Official Full Name** | **Student ID (MTech Applicable)** | **Email** | **Main Contributions** |
|---|---|---|---|
| Sun Wenjing | A0326610N | [e1538441@u.nus.edu](mailto:e1538441@u.nus.edu) | Subject detection and skeleton extraction; dog-pose fine-tuning and quality evaluation; frontend and backend integration. |
| Zhao Jiayi | A0329873M | [e1554200@u.nus.edu](mailto:e1554200@u.nus.edu) | Behaviour and interaction video collection and annotation; dog-behaviour and human–dog interaction model design, training and optimisation; geometric and relation feature engineering; grouped cross-validation and evaluation; video-to-timeline inference pipeline. |

## Project Overview

This project helps pet owners review video recordings without watching every minute. It detects a dog and a person, extracts their skeletons, recognises activities over time, and returns a timestamped timeline with the relevant video intervals for playback.

The prototype is a mobile web application backed by a Python inference pipeline. The current scope is indoor footage with **one principal dog and at most one principal person**. Audio is not used.

## What the system does

- Accepts a video through the mobile web interface or a Python entry point.
- Uses RTMDet-m to locate subjects and separate RTMPose-m branches to extract human COCO-17 and dog AP-10K-17 keypoints.
- Preserves original-image coordinates and detection-box motion, including the dog's movement across the image and the distance between subjects.
- Predicts dog behaviour or human-dog interaction from temporal windows.
- Merges neighbouring predictions into an activity timeline and supports reviewing the corresponding video intervals.

In the default `both` mode, a window with sufficiently detected person and dog produces an **interaction** prediction and suppresses the separate dog-behaviour output. Dog-only windows use the dog-behaviour model. Windows without enough detections are marked accordingly.

| Task | Classes |
|---|---|
| Dog behaviour | Walking/running, standing, sitting, lying/sleeping, eating/drinking |
| Human-dog interaction | No interaction, hand feeding, petting, playing together |

Holding or embracing the dog is included in **petting**. Playing alone is excluded from the final dog-behaviour label set. The hand-feeding label is stored as `hand_feeding` in manifests and inference results.

## Pipeline and selected models

```text
Uploaded video
  → 15 fps frame sampling
  → RTMDet subject boxes and continuity-based subject selection
  → Human and dog RTMPose skeletons
  → Preprocessing and 32-frame recognition inputs
  → Dog-behaviour or human-dog interaction prediction
  → Temporal smoothing and timeline merging
  → Results and video playback
```

| Component | Selected implementation |
|---|---|
| Detection | Official COCO RTMDet-m with project-specific box selection |
| Human pose | Official RTMPose-m, COCO-17 |
| Dog pose | P0: official RTMPose-m AP-10K, retained after comparison with APT-36K fine-tuning |
| Dog behaviour | Motion-aware TCN + 834-dimensional geometry/motion features + two pairwise ExtraTrees specialists; five-fold model average |
| Human-dog interaction | 352-dimensional relation features, ExtraTrees/linear-SVM soft voting, and an auxiliary frozen CLIP ViT-B/32 visual branch |
| Application | FastAPI backend and React/Vite mobile web frontend |

The dog model combines learned temporal patterns with explicit limb, head, paw and trajectory cues. The interaction model measures cross-subject relations such as hand-to-nose distance, arm-to-trunk proximity, contact duration and relative activity. Visual context is auxiliary to the relation branch.

ST-GCN, LSTM and Transformer variants were explored during development; they are **not the released recognition models**. Only the selected recognition configurations and their ensemble components are published.

## Evaluation

The primary recognition results use **grouped five-fold out-of-fold (OOF) evaluation**. Dog-behaviour splits group dog identities and source videos. Interaction splits also group person identities, keeping connected identity groups together. Each development clip receives a prediction from a fold that did not train on that clip.

| Task | OOF clips | OOF Accuracy | OOF Macro-F1 | Small-test clips | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Dog behaviour | 137 | 77.37% | 76.47% | 33 | 78.79% | 78.83% |
| Human-dog interaction | 97 | 77.32% | 76.13% | 24 | 83.33% | 83.30% |

The small test sets provide supporting feasibility results. Use the grouped OOF results as the main comparison, and consult the [model release documentation](pet_recognition_system/checkpoints/best_models/README.md) for the evaluation protocol and benchmark limitations.

Dog-pose quality was evaluated on **36 indoor frames from 28 dogs**, with **547 manually labelled visible joints**. Using RTMDet boxes, P0 achieved 67.8% OKS AP and 85.7% PCK@0.1. APT-36K fine-tuning improved its own validation AP but showed no clear indoor advantage, so P0 was retained. See [Section 6.3 of the report](Final_Report.md#63-dog-pose-estimation-results) for the comparison.

Saved recognition metrics, confusion matrices and split manifests are in [outputs/best_models](pet_recognition_system/outputs/best_models/).

## Repository structure

```text
PRS/
├── frontend/                     React/Vite mobile web application
├── pet_recognition_system/
│   ├── app/                      FastAPI upload and analysis service
│   ├── configs/                  Model and training configuration
│   ├── pet_recognition/          Models, datasets, preprocessing and inference
│   ├── tools/                    Training, evaluation and video prediction entry points
│   ├── checkpoints/best_models/  Selected recognition weights and model documentation
│   ├── outputs/best_models/      Frozen metrics, predictions and split manifests
│   ├── tests/                    Backend and recognition tests
│   ├── ARCHITECTURE.md           Module responsibilities
│   └── MODEL_TRAINING.md         Training and evaluation commands
├── dataset/                      Link to the hosted project data
├── diagrams/                     System and report figures
├── presentation/
│   ├── Group28_Proposal.pptx
│   ├── Group28_Final.pptx
│   └── Group28_Presentation_Assignment.txt
├── proposal/                     Proposal materials
├── prototype/                    Earlier interface design materials
└── Final_Report.md               Final report in Markdown
```

## Quick start: frontend demo

The mock frontend runs without downloading datasets or model weights. It uses sample data and **does not perform model inference**.

```bash
git clone https://github.com/ZHAO-1221/PRS.git
cd PRS/frontend
npm install
npm run dev:mock
```

Open the address printed by Vite. For frontend structure and development commands, see [frontend/README.md](frontend/README.md).

## Run the complete system

### 1. Prepare the Python environment

Use **Python 3.11** and follow the [system setup instructions](pet_recognition_system/README.md#setup), including the CPU OpenMMLab installation for RTMDet. The basic requirements alone do not install the full detector stack.

From `pet_recognition_system/`, install the application dependencies and local package in that environment:

```bash
python -m pip install -r requirements-app.txt
python -m pip install -e .
```

Keep the pinned dependency versions, particularly scikit-learn, consistent with the saved recognition models. The CLI's optional segment export requires FFmpeg on `PATH` or the `imageio-ffmpeg` package.

### 2. Prepare external pose weights

The selected behaviour and interaction weights are included in [checkpoints/best_models](pet_recognition_system/checkpoints/best_models/). The upstream pose-pipeline weights must be provided separately:

```text
<PET_DATA_ROOT>/models/
├── rtmdet-m_coco/
│   └── rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth
├── rtmpose-m_coco_human/
│   └── end2end.onnx
└── rtmpose-m_ap10k/
    └── end2end.onnx
```

Interaction inference also uses the official [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) model. The pipeline prefers a complete local snapshot at `<PET_DATA_ROOT>/models/clip-vit-base-patch32/`; otherwise it downloads the official model on first use and reuses the Hugging Face cache. Prepare the local snapshot or cache before offline use.

### 3. Start the backend and frontend

In a terminal opened at `pet_recognition_system/`, set `PET_DATA_ROOT` to your own data directory and start FastAPI.

PowerShell:

```powershell
$env:PET_DATA_ROOT = "C:\path\to\data"
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Bash:

```bash
export PET_DATA_ROOT=/path/to/data
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

In a second terminal, from `frontend/`:

```bash
npm run dev
```

Vite forwards `/api` requests to the backend on port 8000. Its printed LAN address can also be opened on a phone on the same network. API documentation is available at `http://localhost:8000/docs`.

### 4. Run video prediction directly

The same pipeline can run without the frontend. From `pet_recognition_system/`:

```bash
python tools/predict_video.py /path/to/input.mp4 --task both --data-root /path/to/data
```

The default output folder is `outputs/predictions/<video-name>/`, containing:

- `result.json`: configuration, window predictions and merged timelines.
- `timeline.csv`: start time, end time, activity label and confidence for each segment.

Videos up to eight seconds are classified as one clip. Longer recordings use four-second windows by default. Use `--task dog_behavior` or `--task interaction` to run one task explicitly, and `--export-clips` to export the merged intervals as video files.

Other Python code can call the public entry point:

```python
from tools.predict_video import predict_video

result = predict_video(
    input_path="uploads/example.mp4",
    output_dir="outputs/example",
    data_root="/path/to/data",
    task="both",
)
```

## Data, training and documentation

Project videos, annotations and generated skeletons are hosted in the [PRS Group 28 Hugging Face dataset](https://huggingface.co/datasets/melodyyyyy123/PRS-Group28-data). Raw project data are kept outside this Git repository. Use the frozen manifests in `outputs/best_models/manifests/` to reproduce the published recognition splits.

| Resource | Location |
|---|---|
| Final report | [Final_Report.md](Final_Report.md) |
| Module architecture | [ARCHITECTURE.md](pet_recognition_system/ARCHITECTURE.md) |
| Installation and inference | [System README](pet_recognition_system/README.md) |
| Training and evaluation commands | [MODEL_TRAINING.md](pet_recognition_system/MODEL_TRAINING.md) |
| Selected models and benchmark details | [Model release README](pet_recognition_system/checkpoints/best_models/README.md) |
| Frontend instructions | [Frontend README](frontend/README.md) |
| Proposal slides | [Group28_Proposal.pptx](presentation/Group28_Proposal.pptx) |
| Final slides | [Group28_Final.pptx](presentation/Group28_Final.pptx) |
| Speaker assignment and slide outline | [Group28_Presentation_Assignment.txt](presentation/Group28_Presentation_Assignment.txt) |

Run backend tests from `pet_recognition_system/` with `python -m pytest -q`, and frontend tests from `frontend/` with `npm test` after installing the corresponding dependencies.

## Scope and limitations

The current system selects one principal subject per species and does not resolve general multi-person or multi-dog interactions. Pose estimates and geometric contact cues are affected by viewpoint, occlusion and missed detections. The detector merges several quadruped categories to recover misclassified dogs, so another animal in view can be mistaken for a dog. Predictions describe visible activities, not health or emotional diagnoses.

## Course submission checklist

The team submission is **one ZIP**, containing the runnable system, datasets, final report, code and model files, supporting documents, a **10–15 minute presentation video**, and the slides from both the Proposal and Final presentations. The report should cover tools and techniques, system design and models, performance, and findings and discussion.

Each member also submits a separate **1–2 page individual report** covering personal contributions, useful learning and applications in other contexts, and completes **Peer Review** through the designated system.

**Submission deadline: 31 October 2026.** Submit through **Canvas → Assignments → Practice Module**. The individual report is submitted separately from the team ZIP.
