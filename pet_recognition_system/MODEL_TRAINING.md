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

For dog-behaviour retraining, create one `<dog-skeleton-dir>` containing the
files from both Hugging Face directories `data/skeletons_generated/` and
`data/skeletons_generated_APT/`. Rename each APT file from `<clip_id>.npz` to
`dog_<clip_id>.npz`. Interaction training continues to use
`data/skeletons_generated/` directly.

## Dog behaviour

### 1. Train five motion-aware TCN folds

```bash
python tools/cross_validate_dog_behavior.py \
  --manifest outputs/best_models/manifests/dog_behavior_manifest_5class_v4.csv \
  --skeleton-dir <dog-skeleton-dir> \
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
  --skeleton-dir <dog-skeleton-dir> \
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
  --skeleton-dir <dog-skeleton-dir> \
  --cv-dir work_dirs/dog_behavior_5class_v4_cv \
  --fusion-dir work_dirs/dog_behavior_5class_v5_head_duration_fusion \
  --feature-version head_duration_v1 \
  --output-dir work_dirs/dog_behavior_retrained_test
```

## Human-dog interaction

### 1. Recreate the mixed-v2 split

```bash
python tools/prepare_interaction_mixed_split.py \
  --manifest outputs/best_models/manifests/interaction_manifest.csv \
  --selection outputs/demo_evaluation/interaction/mixed_demo_predictions.csv \
  --output outputs/best_models/manifests/interaction_manifest_mixed_v2.csv
```

### 2. Run five-fold identity-grouped cross-validation

```bash
python tools/cross_validate_interaction_relation.py \
  --manifest outputs/best_models/manifests/interaction_manifest_mixed_v2.csv \
  --skeleton-dir <data>/skeletons_generated \
  --folds 5 \
  --augmentations 0 \
  --seed 28 \
  --classifier soft_vote \
  --max-features sqrt \
  --min-samples-leaf 1 \
  --hand-feeding-weight 1.5 \
  --svm-c 0.005 \
  --svm-weight 0.2 \
  --output-dir work_dirs/interaction_relation_mixed_v2_cv5_noaug
```

### 3. Fit the final five-seed ensemble on all train and validation clips

```bash
python tools/train_final_interaction_ensemble.py \
  --manifest outputs/best_models/manifests/interaction_manifest_mixed_v2.csv \
  --skeleton-dir <data>/skeletons_generated \
  --output-dir work_dirs/interaction_relation_mixed_v2_final \
  --models 5 \
  --seed 28 \
  --classifier soft_vote \
  --max-features sqrt \
  --min-samples-leaf 1 \
  --hand-feeding-weight 1.5 \
  --svm-c 0.005 \
  --svm-weight 0.2
```

### 4. Extract frozen CLIP interaction features

```bash
python tools/extract_interaction_clip_features.py \
  --manifest outputs/best_models/manifests/interaction_manifest_mixed_v2.csv \
  --video-dir <data>/interaction \
  --skeleton-dir <data>/skeletons_generated \
  --output-dir outputs/interaction_clip_features_clip_v2 \
  --frames 8
```

### 5. Train and evaluate the conservative multimodal fusion

The visual branch is selected with grouped OOF predictions. The released model
uses a conservative visual log-probability weight of `0.1`; the relation branch
remains the primary decision source.

```bash
python tools/cross_validate_interaction_rgb_fusion.py \
  --manifest outputs/best_models/manifests/interaction_manifest_mixed_v2.csv \
  --skeleton-dir <data>/skeletons_generated \
  --feature-dir outputs/interaction_clip_features_clip_v2 \
  --base-oof-dir work_dirs/interaction_relation_mixed_v2_cv5_noaug \
  --base-final-dir work_dirs/interaction_relation_mixed_v2_final \
  --output-dir work_dirs/interaction_multimodal_release \
  --deployment-alpha 0.1
```

### 6. Evaluate the frozen final ensemble

```bash
python tools/evaluate_interaction_multimodal.py \
  --model-root checkpoints/best_models/interaction \
  --manifest outputs/best_models/manifests/interaction_manifest_mixed_v2.csv \
  --skeleton-dir <data>/skeletons_generated \
  --feature-dir outputs/interaction_clip_features_clip_v2 \
  --split test \
  --output-dir work_dirs/interaction_multimodal_release_check
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
├── prepare_interaction_mixed_split.py
├── train_final_interaction_ensemble.py
├── extract_interaction_clip_features.py
├── cross_validate_interaction_rgb_fusion.py
└── evaluate_interaction_multimodal.py

pet_recognition/models/
└── skeleton_tcn.py
```

Only the selected best-performing training path is published. Earlier ST-GCN,
LSTM, Transformer, RGB-fusion, hierarchical and ablation experiments are not
part of this release.
