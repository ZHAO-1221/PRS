# Model Training

Training entry points stay in `tools/`; reusable network definitions stay in
`pet_recognition/models/`; training products stay in `work_dirs/`. Frozen
weights selected for sharing are copied to `checkpoints/best_models/`.

The exact split manifests used by the published models are stored in
`outputs/best_models/manifests/`.

## Environment

```bash
pip install -r requirements-training.txt
pip install -e .
```

Set `<skeleton-dir>` to the Hugging Face dataset's
`data/skeletons_generated/` directory.

## Dog behaviour

### 1. Train five motion-aware TCN folds

```bash
python tools/cross_validate_dog_behavior.py \
  --manifest outputs/best_models/manifests/dog_behavior_manifest_5class_v4.csv \
  --skeleton-dir <skeleton-dir> \
  --model motion_aux_tcn \
  --folds 5 \
  --epochs 80 \
  --patience 15 \
  --target-frames 32 \
  --seed-base 28 \
  --output-dir work_dirs/dog_behavior_5class_v4_cv
```

### 2. Train the geometry and pairwise fusion layers

```bash
python tools/cross_validate_dog_5class_fusion.py \
  --cv-dir work_dirs/dog_behavior_5class_v4_cv \
  --skeleton-dir <skeleton-dir> \
  --feature-version head_duration_v1 \
  --folds 5 \
  --seed 28 \
  --output-dir work_dirs/dog_behavior_5class_v5_head_duration_fusion
```

`head_duration_v1` is required to reproduce the published 834-dimensional
feature vector. Later experimental features use `fixed_region_v2` and are not
part of the selected release.

### 3. Evaluate the selected fold ensemble

```bash
python tools/evaluate_dog_5class_ensemble.py \
  --manifest outputs/best_models/manifests/dog_behavior_manifest_5class_v4.csv \
  --skeleton-dir <skeleton-dir> \
  --cv-dir work_dirs/dog_behavior_5class_v4_cv \
  --fusion-dir work_dirs/dog_behavior_5class_v5_head_duration_fusion \
  --feature-version head_duration_v1 \
  --output-dir work_dirs/dog_behavior_retrained_test
```

## Human-dog interaction

### 1. Train five identity-grouped ExtraTrees folds

```bash
python tools/cross_validate_interaction_relation.py \
  --manifest outputs/best_models/manifests/interaction_manifest.csv \
  --skeleton-dir <skeleton-dir> \
  --folds 5 \
  --augmentations 0 \
  --seed 28 \
  --output-dir work_dirs/interaction_relation_cv5_noaug
```

### 2. Evaluate the fold ensemble

```bash
python tools/evaluate_interaction_ensemble.py \
  work_dirs/interaction_relation_cv5_noaug \
  --manifest outputs/best_models/manifests/interaction_manifest.csv \
  --skeleton-dir <skeleton-dir> \
  --split test \
  --output-dir work_dirs/interaction_retrained_test
```

## Relevant files

```text
tools/
├── cross_validate_dog_behavior.py
├── cross_validate_dog_5class_fusion.py
├── evaluate_dog_5class_ensemble.py
├── train_dog_behavior.py
├── cross_validate_dog_joint_specialist.py
├── train_dog_play_specialist.py
├── train_dog_eating_specialist.py
├── train_dog_standing_specialist.py
├── train_dog_walking_specialist.py
├── cross_validate_interaction_relation.py
└── evaluate_interaction_ensemble.py

pet_recognition/models/
└── skeleton_tcn.py
```

Only the selected best-performing training path is published. Earlier ST-GCN,
LSTM, Transformer, RGB-fusion, hierarchical and ablation experiments are not
part of this release.
