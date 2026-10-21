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
- Architecture: five seed ensembles operating on summarized human-dog relation
  features. Each seed uses an 80% ExtraTrees / 20% standardized linear-SVM soft
  vote, and final probabilities are averaged across seeds.
- Training protocol: 5-fold identity-grouped CV for development, followed by
  final fitting on all 97 train/validation clips with no synthetic augmentation.
  Hand-feeding receives a 1.5 tree-class weight; the SVM uses `C=0.005`. Both
  settings and the 20% SVM vote were selected by cross-validation.
- OOF result: 77.32% accuracy, 75.89% macro-F1 on 97 clips.
- Mixed-v2 test result: 17/24 correct, 70.83% accuracy, 69.36% macro-F1.
- Test composition: 6 clips per class and no human, dog or source-video overlap
  with training. Half of this set was selected from previously correct OOF
  examples, so the score is a demonstration benchmark rather than an unbiased
  one-shot estimate.
- Main limitation: hand feeding has 33.33% recall on the mixed-v2 test set.

## Data and privacy

Weights do not contain video files. IDs used during evaluation are anonymous.
The source dataset is maintained separately on Hugging Face.

## Reproducibility

- Release date: 2026-10-22
- Source commit before packaging: `dfacc3f`
- Exact evaluation reports are included under `outputs/best_models/`.

## Weight-file safety

PyTorch and Joblib files use Python serialization. Only load weights obtained
from this trusted repository after verifying `SHA256SUMS.txt`; never load
untrusted replacement files.
