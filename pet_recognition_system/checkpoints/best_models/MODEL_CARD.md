# Model Card

## Scope

These models classify short indoor third-person clips after human and dog
skeleton extraction. They are research prototypes for PRS Group 28, not safety
systems and not veterinary diagnostic tools.

## Dog behaviour model

- Input: AP-10K dog skeleton, shape `T x 17 x 3`, detector-box global motion,
  and a frame mask; inference resamples to `T=32`.
- Classes: `walking_run`, `standing`, `sitting`, `lying_sleeping`,
  `eating_drinking`.
- Architecture: five motion-aware TCN folds; fold probabilities are refined by
  five multiclass geometry models and two pairwise specialist stages per fold.
- Engineered feature version: `head_duration_v1`.
- Training data: 170 valid project clips plus 31 curated external APT skeleton
  clips used only for training.
- Test result: 26/33 correct, 78.79% accuracy, 78.83% macro-F1.
- Main limitation: eating/drinking has 50% test recall.

## Human-dog interaction model

- Input: synchronized COCO-17 human and AP-10K-17 dog skeletons plus frame mask.
- Classes: `no_interaction`, `hand_feeding`, `petting`, `playing_together`.
- Architecture: five ExtraTrees classifiers operating on summarized
  human-dog relation features; probabilities are averaged across folds.
- Training protocol: leakage-safe connected human/dog identity grouping,
  five-fold cross-validation, no synthetic augmentation.
- OOF result: 82.11% accuracy, 82.50% macro-F1 on 123 clips.
- Test result: 12/21 correct, 57.14% accuracy, 57.27% macro-F1.
- Main limitation: substantial domain shift on unseen test clips, especially
  for hand feeding and petting.

## Data and privacy

Weights do not contain video files. IDs used during evaluation are anonymous.
The source dataset is maintained separately on Hugging Face.

## Reproducibility

- Release date: 2026-10-21
- Source commit at packaging: local working tree after `9162c84452295b7db9b97165240ff8b781f01de8`
- Exact evaluation reports are included under `outputs/best_models/`.

## Weight-file safety

PyTorch and Joblib files use Python serialization. Only load weights obtained
from this trusted repository after verifying `SHA256SUMS.txt`; never load
untrusted replacement files.
