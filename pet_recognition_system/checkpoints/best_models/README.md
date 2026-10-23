# Best Skeleton Recognition Models

This checkpoint release contains the frozen fold ensembles selected for the two
active classification tasks. Frozen reports are kept in the repository's
`outputs/best_models/` directory; video clips and generated skeleton arrays
remain in the Hugging Face dataset.

Only the best-performing configuration for each active task is included. The
fold files inside each task directory are components of that single ensemble,
not older model versions. Experimental and superseded checkpoints are excluded.

## Included models

| Task | Model | Classes | Development result | Test result |
|---|---|---:|---:|---:|
| Dog behaviour | Motion-aware TCN + engineered geometry ensemble | 5 | 77.37% accuracy / 76.47% macro-F1 | 78.79% accuracy / 78.83% macro-F1 |
| Human-dog interaction | Relation ensemble + frozen CLIP visual context | 4 | 77.32% OOF accuracy / 76.13% macro-F1 | 83.33% accuracy / 83.30% macro-F1 |

The dog test set has been used as an iterative benchmark, so its score must not
be described as a one-shot unbiased final estimate. The interaction mixed-v2
test set has 24 clips (6 per class); half were selected from previously correct
group-OOF examples, so it is a demonstration benchmark rather than an unbiased
one-shot estimate.
The selected dog model adds 31 curated APT skeleton clips to training only;
validation and the fixed 33-clip test split contain no APT clips.

## Directory layout

```text
pet_recognition_system/
├── checkpoints/best_models/
│   ├── dog_behavior/
│   │   ├── backbone/fold_0..4/best.pt
│   │   ├── fusion/*.joblib
│   │   └── config.json
│   ├── interaction/
│   │   ├── checkpoints/fold_1..5.joblib
│   │   ├── rgb/fold_1..5.joblib
│   │   └── config.json
│   ├── MODEL_CARD.md
│   ├── SOURCE_FILES.txt
│   └── SHA256SUMS.txt
└── outputs/best_models/
    ├── dog_behavior/
    └── interaction/
```

## Requirements

Install the project and training dependencies from the repository root:

```bash
pip install -r requirements-training.txt
pip install -e .
```

The dog model expects generated skeleton NPZ files. The interaction model also
uses eight frozen CLIP frame features from the raw clip. Set `PET_DATA_ROOT` to
the downloaded Hugging Face `data/` directory or pass the data directories
explicitly.

Complete retraining commands are documented in the repository root's
[`MODEL_TRAINING.md`](../../MODEL_TRAINING.md).

## Reproduce dog-behaviour evaluation

```bash
python tools/evaluate_dog_5class_ensemble.py \
  --manifest outputs/best_models/manifests/dog_behavior_manifest_5class_v4.csv \
  --skeleton-dir <data>/skeletons_generated \
  --cv-dir checkpoints/best_models/dog_behavior/backbone \
  --fusion-dir checkpoints/best_models/dog_behavior/fusion \
  --output-dir work_dirs/released_dog_evaluation \
  --feature-version head_duration_v1
```

The `head_duration_v1` feature version is required. It reproduces the released
ensemble's 834-dimensional engineered feature vector.

## Reproduce interaction evaluation

```bash
python tools/evaluate_interaction_multimodal.py \
  --model-root checkpoints/best_models/interaction \
  --manifest outputs/best_models/manifests/interaction_manifest_mixed_v2.csv \
  --skeleton-dir <data>/skeletons_generated \
  --feature-dir <generated-interaction-clip-features> \
  --split test \
  --output-dir work_dirs/released_interaction_evaluation
```

Verify downloaded files with `SHA256SUMS.txt` before use.
