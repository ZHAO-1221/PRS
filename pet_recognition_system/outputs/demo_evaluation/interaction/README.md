# Interaction mixed-v2 selection record

This directory records how the balanced mixed-v2 test IDs were selected. The
active retrained-model reports are under `outputs/best_models/interaction/`.

- Size: 24 clips, with 6 clips per interaction class.
- Easy half: 12 original-training clips correctly classified by their held-out
  group-OOF fold, with 3 clips per class.
- Fixed-test half: 12 clips from the original fixed test set, selected
  deterministically by class and clip ID, with 3 clips per class.
- Intended use: qualitative demonstrations and presentation examples only.

Before retraining, the old OOF/fixed-test predictions gave this selected set
75.00% accuracy and 75.61% macro-F1. That number is only a split-selection
record and is not produced by the active weights. After identity isolation and
retraining, the active weights achieve 70.83% accuracy and 69.36% macro-F1 on
these 24 clips. The earlier fixed-v1 result remains archived separately.
