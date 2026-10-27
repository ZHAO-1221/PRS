# Final Report: Skeleton-Based Dog Behaviour and Human-Dog Interaction Recognition

**Group 28 · PRS Practice Module**

## Table of Contents

- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
  - [1.1 Background and Motivation](#11-background-and-motivation)
  - [1.2 Problem Statement](#12-problem-statement)
  - [1.3 Objectives and Scope](#13-objectives-and-scope)
  - [1.4 Report Organisation](#14-report-organisation)
- [2. Datasets](#2-datasets)
  - [2.1 Data Sources](#21-data-sources)
  - [2.2 Dog Behaviour Clips](#22-dog-behaviour-clips)
  - [2.3 Human-Dog Interaction Clips](#23-human-dog-interaction-clips)
  - [2.4 Dog Pose Data (Hand-Labelled Evaluation Set and APT-36K)](#24-dog-pose-data-hand-labelled-evaluation-set-and-apt-36k)
  - [2.5 Annotation and Data Preparation Pipeline](#25-annotation-and-data-preparation-pipeline)
  - [2.6 Data Splits and Leakage Control](#26-data-splits-and-leakage-control)
- [3. Tools and Techniques](#3-tools-and-techniques)
  - [3.1 Software Stack and Frameworks](#31-software-stack-and-frameworks)
  - [3.2 Object Detection (RTMDet)](#32-object-detection-rtmdet)
  - [3.3 Pose Estimation (RTMPose)](#33-pose-estimation-rtmpose)
  - [3.4 Multi-Target Tracking and Skeleton Extraction](#34-multi-target-tracking-and-skeleton-extraction)
  - [3.5 Skeleton Preprocessing and Normalisation](#35-skeleton-preprocessing-and-normalisation)
  - [3.6 Temporal Modelling (TCN)](#36-temporal-modelling-tcn)
  - [3.7 Feature Engineering and Classical Classifiers](#37-feature-engineering-and-classical-classifiers)
  - [3.8 Ensemble Learning](#38-ensemble-learning)
  - [3.9 Evaluation Methods](#39-evaluation-methods)
- [4. System Design](#4-system-design)
  - [4.1 Overall System Architecture](#41-overall-system-architecture)
  - [4.2 Dual-Species Skeleton Extraction Pipeline](#42-dual-species-skeleton-extraction-pipeline)
  - [4.3 Repository and Module Structure](#43-repository-and-module-structure)
  - [4.4 Frontend Prototype](#44-frontend-prototype)
- [5. Models](#5-models)
  - [5.1 Model One: Dog Pose Estimation](#51-model-one-dog-pose-estimation)
    - [5.1.1 Baseline (P0: RTMPose-m AP-10K)](#511-baseline-p0-rtmpose-m-ap-10k)
    - [5.1.2 Fine-Tuning on APT-36K (P1)](#512-fine-tuning-on-apt-36k-p1)
    - [5.1.3 Further Adaptation (P2)](#513-further-adaptation-p2)
  - [5.2 Model Two: Dog Behaviour Recognition](#52-model-two-dog-behaviour-recognition)
    - [5.2.1 Motion-Aware TCN Backbone](#521-motion-aware-tcn-backbone)
    - [5.2.2 Engineered Geometry Features](#522-engineered-geometry-features)
    - [5.2.3 Geometry and Pairwise Fusion](#523-geometry-and-pairwise-fusion)
    - [5.2.4 Class-Specialist Models](#524-class-specialist-models)
  - [5.3 Model Three: Human-Dog Interaction Recognition](#53-model-three-human-dog-interaction-recognition)
    - [5.3.1 Human-Dog Relation Features](#531-human-dog-relation-features)
    - [5.3.2 ExtraTrees / Linear-SVM Soft-Vote Ensemble](#532-extratrees--linear-svm-soft-vote-ensemble)
    - [5.3.3 Frozen CLIP Visual Branch](#533-frozen-clip-visual-branch)
    - [5.3.4 Probability-Level Fusion](#534-probability-level-fusion)
  - [5.4 Alternative Models Explored](#54-alternative-models-explored)
- [6. System Performance](#6-system-performance)
  - [6.1 Experimental Setup](#61-experimental-setup)
  - [6.2 Evaluation Metrics](#62-evaluation-metrics)
  - [6.3 Dog Pose Estimation Results](#63-dog-pose-estimation-results)
  - [6.4 Dog Behaviour Recognition Results](#64-dog-behaviour-recognition-results)
  - [6.5 Human-Dog Interaction Recognition Results](#65-human-dog-interaction-recognition-results)
  - [6.6 Ablation Studies](#66-ablation-studies)
  - [6.7 Comparison with Alternative Models](#67-comparison-with-alternative-models)
  - [6.8 Runtime and Deployment Considerations](#68-runtime-and-deployment-considerations)
- [7. Findings and Discussion](#7-findings-and-discussion)
  - [7.1 Key Findings](#71-key-findings)
  - [7.2 Error Analysis](#72-error-analysis)
  - [7.3 Impact of Detection and Tracking Quality](#73-impact-of-detection-and-tracking-quality)
  - [7.4 Data Quality, Size and Leakage](#74-data-quality-size-and-leakage)
  - [7.5 Limitations](#75-limitations)
  - [7.6 Future Work](#76-future-work)
- [8. Conclusion](#8-conclusion)
- [References](#references)
- [Appendix](#appendix)
  - [A. Installation and User Guide](#a-installation-and-user-guide)
  - [B. Model Training and Reproduction Commands](#b-model-training-and-reproduction-commands)
  - [C. Dataset and Model File Listing](#c-dataset-and-model-file-listing)
  - [D. Additional Results](#d-additional-results)

---

## Abstract

## 1. Introduction

### 1.1 Background and Motivation

### 1.2 Problem Statement

### 1.3 Objectives and Scope

### 1.4 Report Organisation

## 2. Datasets

### 2.1 Data Sources

The project combines public pose data with a project-curated video collection. For dog-pose adaptation, the dog subset of APT-36K is used while retaining its official training and validation separation. It provides animal pose supervision that is independent of the project videos and is converted to the 17-keypoint convention used by the dog-pose branch. The hand-labelled pose evaluation images are kept separate from behaviour and interaction labels and are described in Section 2.4.

The dog-behaviour and human--dog interaction datasets were assembled from project-contributed clips and publicly available research-use videos collected from sources including YouTube, Pexels, Mixkit, InterPet and SmartHomeBench. Selection prioritised indoor, third-person and relatively stable-camera footage in which the principal subjects were visible. A limited number of outdoor or partially occluded clips were retained when the action boundary remained unambiguous, providing moderate viewpoint and scene variation. Audio is not used by the current system.

Every original video is assigned a `source_video_id`, and each dog and person is represented by an anonymous `dog_id` or `person_id`. Source references and quality notes are retained in the manifests so that each short clip can be traced to its original video without storing personal names.

### 2.2 Dog Behaviour Clips

### 2.3 Human-Dog Interaction Clips

The interaction dataset contains **121 valid clips** drawn from **80 source videos**, with 69 anonymous dog identifiers and 69 anonymous person identifiers. Each clip contains one principal person and one principal dog and has one stable interaction label. The clips range from 1.679 to 10.239 seconds, with a mean duration of 4.830 seconds. Label transitions are cut into separate clips rather than assigning multiple sequential actions to one sample.

The four interaction classes are defined as follows:

- **No interaction:** the person and dog are both present, but there is no sustained physical contact, hand feeding or mutually engaged play. Speech, looking or gestures without contact remain in this class.
- **Hand feeding:** food or a reward is transferred from the person's hand, or from a hand-held container, towards the dog's mouth. Independent eating from a bowl is not labelled as human--dog interaction.
- **Petting:** the person's hand maintains contact with the dog's head or trunk and performs a small stroking movement. Sustained holding or embracing is also included, whereas incidental contact is excluded.
- **Playing together:** the person and dog are both actively engaged, for example through chasing, toy tugging or repeated large coordinated movements. A dog playing alone is not included in this class.

The final fixed split is shown below. The 24-clip test set is balanced with six clips per class; the remaining 97 clips form the development set used for grouped five-fold evaluation.

| Interaction class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| No interaction | 23 | 4 | 6 | 33 |
| Hand feeding | 18 | 5 | 6 | 29 |
| Petting | 19 | 6 | 6 | 31 |
| Playing together | 15 | 7 | 6 | 28 |
| **Total** | **75** | **22** | **24** | **121** |

### 2.4 Dog Pose Data (Hand-Labelled Evaluation Set and APT-36K)

Model One uses two pose datasets. Neither is used to train the behaviour or interaction classifiers.

**APT-36K dog subset.** The dog annotations of APT-36K provide the training data for the fine-tuned pose variant (Section 5.1.2). The subset contains 1,193 frames and 2,245 dog annotations from 27 source videos, all at 1920×1080 resolution. A total of 180 annotations contain no labelled keypoint and are removed, leaving 2,065 dog instances. The data are split by source video, so that frames from one video never appear in both partitions. APT-36K was chosen instead of AP-10K because the baseline model had already been trained on AP-10K. Its keypoints are either labelled or unlabelled; there is no separate state for a labelled but occluded joint.

| APT-36K split | Source videos | Images | Dog instances |
|---|---:|---:|---:|
| Training | 24 | 983 | 1,691 |
| Validation | 3 | 209 | 374 |

**Indoor evaluation set.** Model selection is based on 36 frames taken from the project's indoor home-camera clips. The frames come from 28 dogs in 28 source videos and contain one dog each. They were selected so that each of the six behaviour labels in use at selection time (walking/running, standing, sitting, lying/sleeping, eating/drinking and playing alone) contributes six frames. Playing alone was later removed from Model Two because it could not be labelled consistently; this does not affect the pose evaluation, which never uses behaviour labels. The grouping is used only to break pose accuracy down by posture.

All 17 AP-10K keypoints were labelled by hand by one annotator. Only visible keypoints whose position could be determined were labelled; joints hidden by the body or furniture, or outside the image, were left unlabelled rather than guessed, and they are excluded from every score. This follows the AP-10K convention and allows a direct comparison with the official AP-10K results in Section 6.3. Severely blurred frames, in which a human annotator could not locate the joints, were not included.

| Item | Value |
|---|---:|
| Frames | 36 (6 per behaviour label) |
| Dogs / source videos | 28 / 28 |
| Keypoint slots (36 × 17) | 612 |
| Labelled visible keypoints | 547 (89%) |
| Unlabelled (occluded or undeterminable) | 65 (11%) |
| Most frequently unlabelled | right knee and right back paw (10 each), left back paw (8), left knee (6) |

None of these frames, and none of their source videos, are used to train any pose model.

### 2.5 Annotation and Data Preparation Pipeline

The dog-behaviour data are prepared in six steps (Figure 2-1), from online videos to the skeleton sequences consumed by Model Two. The interaction clips pass through the same skeleton-extraction steps.

1. **Video collection.** Videos were searched for on public platforms such as YouTube and Bilibili and in research collections such as SmartHomeBench. Footage resembling a home camera was preferred: a fixed, usually elevated indoor viewpoint, a stable camera and a clearly visible dog. Each original video receives a `source_video_id` and each dog an anonymous `dog_id`.
2. **Behaviour clipping.** Each video was watched and cut by hand into short clips in which the dog shows one stable behaviour. Most clips last 5--8 seconds (140 of the 170 valid project clips; median 5.0 s). When posture and activity conflict, the clip is labelled by posture; for example, a dog chewing a toy while lying down is labelled lying/sleeping. Start and end times are recorded in a manifest so that every clip can be re-cut from its source. The actual length of every clip file was checked with `ffprobe` against its labelled interval, and clips whose file ran more than one second longer than the label were re-cut from the original video.
3. **Grouped split.** Clips are assigned to partitions by dog and source video before any frame is sampled (Section 2.6).
4. **15 fps sampling.** Every clip is resampled to 15 frames per second by keeping the first frame of each 1/15-second interval. A common frame rate makes clips recorded at 24--60 fps comparable in time.
5. **Skeleton extraction with Model One.** RTMDet and the tracking rule select the dog (and, for interaction clips, the person) in each frame, and RTMPose estimates 17 keypoints inside the selected box (Sections 3.2--3.4). Annotated skeleton videos of a sample of clips were reviewed by eye, and the tracking rule was revised whenever a systematic failure was found.
6. **Skeleton sequences.** Each clip is stored as one \(2\times T\times17\times3\) skeleton array with the selected boxes, which is handed to Model Two (dog stream only) and Model Three (both streams).

The hand-labelled pose evaluation set (Section 2.4) branches off after the split. Three to six candidate frames per clip were chosen automatically to cover different moments of the clip, ranked by sharpness, dog size and truncation, and then reduced by within-clip similarity. Dog boxes and the 17 keypoints were labelled in LabelMe, with model predictions used only as a starting point that the annotator corrected by hand for every frame. The final 36 evaluation frames were selected from these labelled frames.

![Data preparation pipeline from online videos to skeleton sequences](diagrams/report/figure_2_1_data_preparation_pipeline.png)

*Figure 2-1. Data preparation pipeline. Online home-camera videos are cut by hand into single-behaviour clips, split by dog and source video, sampled at 15 fps and converted by Model One into two-stream skeleton sequences for Models Two and Three. The lower branch shows how frames from the same clips were selected and hand-labelled to form the pose evaluation set used to choose the dog pose weights.*

### 2.6 Data Splits and Leakage Control

The datasets are split at subject and source-video level rather than by individual frames or randomly selected clips. All clips derived from the same `source_video_id` remain in one partition. Dog and person identifiers are also checked jointly: if clips are connected through a shared dog or person, the complete connected identity group is assigned to the same partition. This prevents visually similar neighbouring clips, repeated backgrounds and repeated subjects from appearing on both sides of an evaluation boundary.

For the interaction dataset, the fixed split contains 75 training, 22 validation and 24 test clips. There is no overlap in `dog_id`, `person_id` or `source_video_id` between these partitions. Model development combines the 75 training and 22 validation clips into a 97-clip development set. Stratified grouped five-fold cross-validation is then applied to 47 connected human--dog identity components. In each round, four folds train the model and the remaining fold produces predictions for clips that were not used to fit that fold. Concatenating the five validation predictions yields one out-of-fold prediction for every development clip.

The final interaction ensemble is fitted on all 97 development clips, while the 24-clip test set is excluded from weight fitting and reported only as supporting evidence. Video cutting, frame sampling, pose extraction, feature generation and augmentation inherit the previously assigned split; they are not performed before grouping. The same principle is used for dog-behaviour data, with `dog_id` and `source_video_id` acting as the primary grouping keys.

## 3. Tools and Techniques

### 3.1 Software Stack and Frameworks

The system is divided into a Python model and back-end stack and a JavaScript mobile web front end. All models run on the CPU, so the complete system can be used on an ordinary laptop without a GPU.

| Layer | Tools | Role |
|---|---|---|
| Language and environment | Python 3.11 (Conda) | Inference, evaluation, feature extraction and the back end |
| Pose training | PyTorch 2.1, MMEngine, MMCV 2.1, MMDetection 3.2, MMPose 1.3 | Fine-tuning RTMPose on APT-36K (P1) and exporting checkpoints |
| Inference | ONNX Runtime, OpenCV, NumPy | RTMDet and RTMPose inference, video decoding and 15 fps sampling |
| Recognition models | PyTorch, scikit-learn 1.9, joblib | Motion-aware TCN, ExtraTrees, linear SVM, PCA and logistic regression; saved ensembles |
| Visual features | Hugging Face Transformers | Frozen CLIP ViT-B/32 encoder for the interaction visual branch |
| Video tools | FFmpeg / ffprobe | Clip cutting, re-encoding and length checks |
| Annotation | LabelMe | Dog boxes and keypoints for the pose evaluation set |
| Back end | FastAPI, Uvicorn | REST API for uploading clips, tracking analysis jobs and returning results |
| Front end | React 18, React Router, Vite 5 | Mobile web application |
| Testing | pytest, Vitest | Back-end and model tests; front-end summary-logic tests |
| Plots | Matplotlib | Report figures |

**Back end.** The FastAPI service stores each uploaded video together with its recording time and camera name, and analyses it in a background worker. Analysis runs one clip at a time because CPU inference is heavy. Each analysis job reports four stages (detection, keypoints, behaviour and interaction) so that the client can show progress. The main endpoints upload a clip, query a job, list clips within a date range, and return one clip's result and cover image. Results are written as JSON files next to each clip, so no separate database is required.

**Front end.** The mobile web application is built with React and Vite. Its pages cover the daily story, uploading, analysis progress, the result of one clip and a weekly summary. All data access goes through one API module; a mock mode serves the same pages from sample data, so that the interface can be developed and demonstrated without running the models. During development, Vite forwards `/api` requests to the FastAPI service, and a phone on the same network can open the application directly.

The P1 fine-tuning used an earlier Python 3.10 environment with the same OpenMMLab versions. MMCV provides no Windows build for the CUDA 12.8 runtime required by the available RTX 5060 GPU, so training also ran on the CPU.

### 3.2 Object Detection (RTMDet)

RTMDet is a one-stage real-time object detector built on a CSPNeXt backbone. The project uses RTMDet-m with the official COCO weights and does not retrain it. Of the 80 COCO categories, only the person and dog outputs are retained.

A COCO-trained detector frequently assigns indoor dogs to another quadruped category. On the indoor evaluation set, the dog was localised correctly in every frame, but in 9 of the 36 frames another category scored higher than "dog" (eight as cat and one as cow). All quadruped categories (cat, dog, horse, sheep, cow, bear and similar) are therefore merged into a single dog class, and the merged box with the highest score is used. A box must score at least 0.3 to start a new target; lower-scoring boxes are accepted only to continue an existing track (Section 3.4).

With this post-processing, a dog was found in all 36 evaluation frames. Compared with the hand-drawn boxes, the detected dog boxes have a median IoU of 0.83 and a minimum of 0.63. The merging rule assumes that the dog is the only animal in view: a real cat in the same room would also be treated as a dog.

### 3.3 Pose Estimation (RTMPose)

RTMPose is a top-down pose estimator: the image region inside a detection box is cropped and resized, and the model predicts the keypoints of the single subject in that crop. RTMPose-m uses a CSPNeXt-m backbone and an RTMCC head. Instead of regressing a heatmap, it applies SimCC, which treats the \(x\)- and \(y\)-coordinate of every keypoint as two separate classification problems over sub-pixel bins (split ratio 2.0). This keeps the model light enough for CPU inference.

Two RTMPose-m models are used:

| Branch | Training data | Input size | Keypoints |
|---|---|---|---|
| Human | COCO | 256×192 | COCO-17 |
| Dog | AP-10K | 256×256 | AP-10K-17: eyes, nose, neck, tail root, and shoulder--elbow--front paw and hip--knee--back paw on each side |

The confidence of a keypoint is the peak value of its SimCC distribution. It is not a probability and can exceed 1 (up to about 1.7) or fall slightly below 0. Confidence is therefore clipped to \([0.001, 1]\), so that a value of 0 always means that the subject was not detected. Both models are exported to ONNX and run with ONNX Runtime on the CPU.

Dog pose accuracy is measured with Object Keypoint Similarity (OKS), the standard COCO and AP-10K metric. For one dog,

\[
\mathrm{OKS}=\frac{\sum_{i}\exp\left(-\dfrac{d_i^{2}}{2s^{2}k_i^{2}}\right)\delta(v_i>0)}{\sum_{i}\delta(v_i>0)},
\]

where \(d_i\) is the distance between the predicted and labelled position of keypoint \(i\), \(s^2\) is the object area (the area of the hand-drawn box, since no segmentation is available), \(k_i\) is the AP-10K per-keypoint constant, and \(\delta(v_i>0)\) restricts the sum to labelled keypoints. An OKS of 1 means perfect agreement. OKS AP is the COCO-style average precision over OKS thresholds from 0.50 to 0.95; AP50 and AP75 use single thresholds. The Percentage of Correct Keypoints, PCK@\(\alpha\), is the proportion of labelled keypoints whose error is no more than \(\alpha\) times the long side of the hand-drawn box; \(\alpha=0.05\) and \(0.1\) are reported.

### 3.4 Multi-Target Tracking and Skeleton Extraction

Every clip is sampled at 15 fps. RTMDet is first run on the whole clip; the person and the dog are then tracked separately, and RTMPose is applied inside the selected box in each frame. The output of a clip is

\[
\mathbf{S}\in\mathbb{R}^{2\times T\times17\times3},
\]

where stream 0 is the person (COCO-17), stream 1 is the dog (AP-10K-17), \(T\) is the number of frames, and each joint stores its \(x\)- and \(y\)-coordinate in pixels and its confidence. A frame in which the target is not found is filled with zeros. The selected box and its detection score are stored alongside each frame. Dog-behaviour clips export only the dog stream: in an early batch export, 30 of 93 dog-only clips contained false "person" tracks on static objects such as photographs, coats and picture-in-picture overlays.

A frame can contain more than one candidate box, and a dog lying down often receives a low detection score. The box-selection rule must therefore keep the same subject across the clip without discarding low-confidence frames of that subject. The final rule is summarised below.

| Rule | Condition | Purpose |
|---|---|---|
| Starting frame | The highest-scoring box in the clip with score ≥ 0.3; if none exists, the highest-scoring box with score ≥ 0.1; otherwise the target is absent from the clip | Start from the most reliable detection |
| Bidirectional tracking | Track forwards to the end and backwards to the start of the clip from the starting frame | Recover frames before the first confident detection |
| Normal continuation | Score ≥ 0.3 and IoU > 0 with the previous box; the box with the largest IoU is chosen | Follow the same subject |
| Low-score continuation | 0.1 ≤ score < 0.3 and IoU ≥ 0.3 with the previous box | Keep lying, distant or partly occluded dogs |
| Track loss | No box accepted for 3 consecutive frames (0.2 s) | Allow a different box to take over |
| Recovery | The last position is remembered and the continuation rules keep being applied around it; after a loss, a box with score ≥ 0.3 elsewhere may also take over | Re-attach the subject after a short occlusion |

The rule was developed in six versions, each motivated by failures found by visually reviewing extracted skeletons:

| Version | Change | Problem addressed |
|---|---|---|
| 1 | Highest-scoring box in every frame | — |
| 2 | Highest-scoring box in the first frame, then the box overlapping the previous one most, falling back to the highest score | Version 1 switched between dogs in multi-dog frames |
| 3 | Overlap only, with no fallback to score | Version 2 could still jump to another animal |
| 4 | Low-score continuation and re-detection after 3 lost frames | Lying dogs scored 0.1--0.3 and were dropped |
| 5 | Whole-clip detection, bidirectional tracking, fallback start at the clip's best box, low-score IoU relaxed from 0.5 to 0.3 | Frames before the first confident box were lost; some lying dogs never reached 0.3 |
| 6 (final) | Remember the last position after a track loss | Dogs returning from behind a person with a low score were not re-attached |

The changes recovered most of the frames in which the detector had found the dog but with a low score. In a human--dog play clip, the dog was detected in 55 of 90 frames with version 3 and in 78 with version 4. A night-time clip of a curled-up sleeping dog rose from 78% to 100% of frames with bidirectional tracking, and a distant dog lying on its back rose from 0% (no box ever reached 0.3) to 78% with version 5 and 90% with version 6. In the first complete export, 12 clips had their target in fewer than 80% of frames; version 6 reduced this number to nine. The remaining cases are mostly genuine absences (the dog leaves the view or is fully hidden by a person), corrupted video frames, or boxes that cover only the dog's head. One night-time infrared clip, in which the detector scored the dog at only 0.04--0.3, produced unusable skeletons and was excluded from the dataset.

![Low-score dog boxes recovered by the tracking rule](diagrams/report/figure_3_1_skeleton_tracking_low_score.jpg)

*Figure 3-1. Four frames from a human--dog play clip in which the dog lies on the floor. The person (blue) and dog (orange) are shown with their boxes and skeletons; each title gives the dog-box detection score. All four dog boxes score between 0.11 and 0.27 and would be discarded by a fixed 0.3 threshold; the low-score continuation rule keeps them because they overlap the dog box of the previous frame.*

Detection and pose estimation therefore remain the main source of missing data. Frames in which the detector produces no usable box cannot be recovered by any selection rule, and lying/sleeping is the weakest posture in both detection and pose accuracy (Section 6.3). The downstream models must therefore tolerate zero-filled frames, which they do through interpolation, frame masks and confidence-weighted pooling.

### 3.5 Skeleton Preprocessing and Normalisation

### 3.6 Temporal Modelling (TCN)

Temporal Convolutional Networks (TCNs) model a sequence using one-dimensional convolutions along the temporal axis. In comparison with a recurrent model, all time positions within a layer can be processed in parallel. Dilated convolution enlarges the temporal receptive field without reducing sequence length, while residual connections support the training of deeper temporal models. These properties make a TCN suitable for recognising actions from short skeleton sequences, where both frame-to-frame joint motion and longer-duration posture patterns are important.

For a residual temporal block, the output can be expressed as

\[
\mathbf{H}_{l+1}=\operatorname{ReLU}\left(\mathbf{H}_{l}+F_l(\mathbf{H}_{l})\right),
\]

where \(F_l\) contains temporal convolutions, normalisation, activation and dropout. The dilations used in this project increase across successive blocks, allowing the model to combine local motion with information from a wider temporal range.

### 3.7 Feature Engineering and Classical Classifiers

Deep temporal features are complemented by explicit geometric and motion descriptors. These descriptors include joint positions, joint velocities, body proportions, head orientation, detection-box displacement, movement duration and statistics computed over a complete clip. Such features are useful in a small-data setting because they directly encode domain-relevant cues, such as prolonged head lowering during eating or sustained hand-to-dog proximity during petting.

The project uses Extremely Randomised Trees (ExtraTrees), linear Support Vector Machines (SVMs) and logistic regression for different feature branches. ExtraTrees averages a large collection of randomised decision trees and can represent non-linear thresholds in distance, duration and motion features. A linear SVM provides a complementary global decision boundary after feature standardisation. For high-dimensional visual features, Principal Component Analysis (PCA) is applied before logistic regression to reduce dimensionality and limit overfitting.

### 3.8 Ensemble Learning

Ensembling is used at three levels. First, classifier probabilities from complementary feature branches are combined by weighted soft voting or weighted log-probability fusion. Second, class-pair correction models refine decisions between selected confusable classes. Third, predictions from five cross-validation folds or five random-seed models are averaged at inference time. The saved fold weights therefore form one ensemble model rather than five separate model designs.

ExtraTrees is itself a **homogeneous ensemble**, because it combines many classifiers of the same type: randomised decision trees. Combining ExtraTrees with a linear SVM creates a **heterogeneous ensemble classifier**, since the two component models have different assumptions and decision functions. The complete interaction system further combines skeleton-relation and CLIP visual features; it can therefore be described more precisely as a heterogeneous ensemble with multi-branch feature fusion rather than simply as a hybrid machine-learning model.

The combination is also motivated by the bias--variance trade-off. Individual decision trees can learn non-linear thresholds and feature interactions but may have high variance on a small dataset. ExtraTrees reduces this variance through feature/threshold randomisation and probability averaging across 500 trees, potentially at the cost of a small increase in bias. In contrast, the regularised linear SVM has a simpler and more stable decision boundary but can have higher bias when the true class boundary is non-linear. Weighted soft voting is intended to retain the non-linear capacity of ExtraTrees while using the SVM as a stable complementary classifier. This is a design motivation rather than proof that variance has been reduced; the claim must be supported by controlled component comparisons.

### 3.9 Evaluation Methods

Model development uses five-fold grouped cross-validation. Four folds are used for training and the remaining fold for validation, so each development sample receives one prediction from a model that was not trained on that sample. Combining these predictions produces the out-of-fold (OOF) result. OOF Accuracy, Macro-F1 and the confusion matrix are the primary evidence used for model comparison.

A small held-out test set is also evaluated after model development. Because the number of test clips is limited, its results are treated as supporting evidence that the end-to-end approach is feasible rather than as the main estimate of generalisation performance.

## 4. System Design

### 4.1 Overall System Architecture

### 4.2 Dual-Species Skeleton Extraction Pipeline

The skeleton extraction stage converts every video clip into the two-stream skeleton tensor used by both recognition models. It is shared by training and inference, so the training skeletons and the skeletons seen by the deployed system are produced in exactly the same way. The pipeline contains five steps:

1. **Frame sampling.** The clip is decoded and sampled at 15 fps.
2. **Whole-clip detection.** RTMDet-m is applied to every frame. Only person boxes and merged quadruped boxes are kept (Section 3.2).
3. **Per-species tracking.** One person track and one dog track are selected with the bidirectional rule of Section 3.4. Dog-behaviour clips track only the dog.
4. **Pose estimation.** The human RTMPose-m model (COCO-17) is applied in the person box and the dog RTMPose-m model (AP-10K-17) in the dog box; confidences are clipped to \([0.001, 1]\).
5. **Output.** The two streams are stored together as one \(2\times T\times17\times3\) array, with zeros for frames in which a target was not found.

The dog-behaviour model reads stream 1 only, while the interaction model reads both streams. Because the human branch uses the official model unchanged, all model-specific work in this stage concerns the dog detector post-processing, the tracking rule and the choice of dog pose weights (Section 5.1).

### 4.3 Repository and Module Structure

### 4.4 Frontend Prototype

## 5. Models

### 5.1 Model One: Dog Pose Estimation

Model One produces the dog skeletons used by both recognition models. The human branch uses the official RTMPose-m COCO model without modification; only the dog branch was considered for adaptation, because public animal pose data contain few indoor home-camera scenes. Three variants were planned:

| Variant | Dog pose weights | Status |
|---|---|---|
| P0 | Official RTMPose-m AP-10K, no further training | Evaluated; **selected** |
| P1 | P0 fine-tuned on the APT-36K dog subset | Trained and evaluated |
| P2 | Fine-tuned on hand-labelled indoor frames | Not carried out |

The variants are compared on the 36-frame indoor evaluation set using the real pipeline, in which the pose model receives RTMDet boxes rather than hand-drawn boxes (Section 6.3).

#### 5.1.1 Baseline (P0: RTMPose-m AP-10K)

P0 is the official MMPose RTMPose-m checkpoint for AP-10K, pre-trained on AIC and COCO and then trained for 210 epochs on AP-10K, which covers 54 animal species. Its published results on the AP-10K test set are 0.722 AP, 0.939 AP50 and 0.788 AP75; for large animals the AP is 0.728. Before any fine-tuning, P0 reaches 0.775 AP (0.940 AP50, 0.882 AP75) on the APT-36K dog validation split. P0 is used without any training and serves both as the baseline and as the starting point for P1.

#### 5.1.2 Fine-Tuning on APT-36K (P1)

P1 continues training P0 on the APT-36K dog training split. The model structure, the KLDiscretLoss of the official configuration and the data augmentation are unchanged, and no layer is frozen. Because training starts from converged weights, the learning rate is one tenth of the official value and the schedule is much shorter.

| Setting | P1 | Official AP-10K configuration |
|---|---|---|
| Epochs | 40 planned; **stopped manually after 30** | 210 |
| Learning rate | 4×10⁻⁴ | 4×10⁻³ |
| Schedule | Linear warm-up for 50 iterations; cosine decay from epoch 20 | Warm-up for 1,000 iterations; cosine decay in the second half |
| Optimiser | AdamW, weight decay 0.05 | Same |
| Batch size | 64 (27 iterations per epoch) | 64 × 8 GPUs |
| Augmentation | Flip, half-body, scale 0.6--1.4, rotation ±80°, HSV jitter, blur, random occlusion | Same |
| Weight averaging | EMA | Same |
| Checkpoint selection | Highest AP on the APT-36K validation split, evaluated every epoch | Evaluated every 10 epochs |

The configuration planned a lighter augmentation stage for epochs 31--40; it was not reached because training was stopped after epoch 30. Training ran on a laptop CPU (Intel Core Ultra 7 255H) and took about 67 minutes for 30 epochs, because no Windows MMCV build supported the CUDA 12.8 runtime required by the available RTX 5060 GPU.

On the APT-36K validation split, AP fell to 0.753 after the first epoch, exceeded P0 from epoch 3 and fluctuated between 0.80 and 0.82 after epoch 10 (Figure 6-1). The best value, 0.817, was reached at epoch 19 and this checkpoint is used as P1. The training loss kept decreasing from 0.458 to 0.277 until epoch 30, while validation AP no longer improved after about epoch 10; the later epochs therefore mainly fitted the APT-36K training videos. Since the validation split contains only three videos, the choice of epoch 19 over neighbouring epochs is partly due to chance.

#### 5.1.3 Further Adaptation (P2)

P2 was intended to fine-tune the dog model on hand-labelled indoor frames from the project's own clips. It was not carried out, for three reasons. First, P1 did not improve indoor accuracy (Section 6.3), so additional data from a different public source was not the limiting factor. Second, P0 already performs close to its own benchmark on indoor dogs: its AP with hand-drawn boxes, 0.700, is within the confidence interval of the official large-animal result. Third, the evaluation frames now come from 28 source videos; a fair P2 would require a new video-level split in which none of these videos is used for training, leaving few videos for training.

P0 skeletons were therefore exported for the downstream models. P2 would become worthwhile if the behaviour or interaction models were shown to be limited by skeleton quality, particularly for lying dogs, which are the weakest posture for both detection and pose estimation.

### 5.2 Model Two: Dog Behaviour Recognition

The dog-behaviour model receives a temporal sequence of AP-10K 17-joint dog skeletons and predicts one of five classes: walking/running, standing, sitting, lying/sleeping, and eating/drinking. Every clip is converted to \(T=32\) frames. Each joint contains an \(x\)-coordinate, a \(y\)-coordinate and a confidence score, giving a basic input tensor

\[
\mathbf{X}\in\mathbb{R}^{32\times17\times3}.
\]

Missing keypoints are temporally interpolated where possible, coordinates are normalised using image and body scale, and a frame mask distinguishes observed frames from padded positions. Unlike a representation based only on frame-wise centred skeletons, the proposed input retains detection-box motion so that movement across the image is not discarded.

#### 5.2.1 Motion-Aware TCN Backbone

At each time position, the network receives three groups of features:

1. 51 pose values from the 17 joints and their confidence scores;
2. 51 first-order temporal differences describing joint motion; and
3. six global motion values derived from the dog detection box, including centre displacement and scale change.

The resulting 108-dimensional vector is projected to 64 channels by a \(1\times1\) convolution. It then passes through three residual TCN blocks with kernel size 3 and dilation rates 1, 2 and 4. All temporal convolutions use stride 1 and suitable padding; therefore, the sequence remains 32 frames long throughout the backbone rather than being temporally downsampled.

The frame-level features are aggregated by confidence-weighted temporal pooling:

\[
\mathbf{h}=\frac{\sum_{t=1}^{T}w_t\mathbf{H}_t}
{\sum_{t=1}^{T}w_t+\epsilon},
\]

where \(w_t\) is the mean confidence of the 17 joints in frame \(t\). During training, an auxiliary moving-versus-static head encourages the backbone to distinguish high-motion behaviours from stable postures. Only the five-class behaviour head is used during inference.

![Architecture of the motion-aware TCN for dog-behaviour recognition](diagrams/report/figure_5_1_motion_aware_tcn.png)

*Figure 5-1. Architecture of the motion-aware TCN. Each 32-frame sequence contains 108 features per frame: 51 pose-coordinate channels, 51 joint-velocity channels and six detection-box/global-motion channels. A \(1\times1\) temporal projection maps the input to 64 channels, followed by three residual temporal blocks with dilation rates 1, 2 and 4. Confidence-weighted temporal pooling produces a 64-dimensional clip representation for five-class prediction. The right-hand panels detail the residual block and the increasing temporal receptive field.*

#### 5.2.2 Engineered Geometry Features

The dataset is relatively small, and several classes differ through cues that are difficult to learn reliably from the TCN alone. The system therefore constructs an additional 834-dimensional clip-level vector from the normalised AP-10K skeleton and the six global-motion channels. The vector contains four concatenated feature blocks:

| Feature block | Dimension | Main information |
|---|---:|---|
| General motion summary | 160 | Detection-box translation and scale change; whole-body and per-joint motion |
| Standing-oriented block | 230 | General motion plus paw support, limb extension, trunk orientation and stillness |
| Eating-oriented block | 262 | General motion plus head-lowering geometry, head motion and low-head duration |
| Walking-oriented block | 182 | General motion plus path persistence, speed, acceleration and paw dynamics |
| **Concatenated vector** | **834** | Input to the engineered-feature classifier |

The feature blocks were originally designed for class-specific specialists and were later concatenated without changing their interfaces. Consequently, the 160-dimensional general-motion summary is present in the general block and is repeated inside the three class-oriented blocks. The 834 dimensions therefore represent the implemented concatenated vector rather than 834 unique physical quantities.

For any frame-level signal \(q_t\), the main temporal summary is

\[
S(q)=\left[\operatorname{mean}(q),\operatorname{std}(q),
\min(q),\max(q),Q_{25}(q),Q_{50}(q),Q_{75}(q)\right].
\]

The **160-dimensional general-motion summary** contains ten frame-level signals: the two-dimensional displacement of the detection-box centre from the first frame, its two-dimensional velocity, relative log-scale, scale velocity, displacement magnitude, centre speed, mean joint speed and maximum joint speed. Applying the seven statistics above gives 70 values. Five additional clip-level quantities describe accumulated centre and joint motion and the horizontal and vertical movement ranges. Finally, each of the 17 joints contributes its mean speed, maximum speed, 90th-percentile speed, spatial range and mean confidence, giving another \(17\times5=85\) values. The total is therefore \(70+5+85=160\).

**(a) Standing-oriented geometry.** The system uses the four AP-10K limb chains: shoulder--elbow--front paw and hip--knee--rear paw on the left and right sides. For each leg, an extension ratio is calculated as

\[
r_{\mathrm{leg}}=
\frac{\lVert\mathbf{p}_{\mathrm{proximal}}-\mathbf{p}_{\mathrm{paw}}\rVert}
{\lVert\mathbf{p}_{\mathrm{proximal}}-\mathbf{p}_{\mathrm{middle}}\rVert+
 \lVert\mathbf{p}_{\mathrm{middle}}-\mathbf{p}_{\mathrm{paw}}\rVert}.
\]

The model also measures the vertical spread of the four paws, their position below the neck/tail-root body axis, the horizontal orientation of the neck-to-tail vector, visible-paw confidence, detection-box centre speed, and mean and maximum joint speed. These ten frame-level signals are each summarised by \(S(\cdot)\), adding 70 values to the 160-dimensional general summary and producing the 230-dimensional standing block. A standing sequence is therefore represented by extended supporting limbs, relatively aligned paws, a near-horizontal trunk and low motion, rather than by a single static pose.

**(b) Eating-oriented geometry.** This block focuses on the nose (joint 2), neck (3), tail root (4), shoulders (5 and 8), front paws (7 and 10) and rear paws (13 and 16). Its main distances are the nose-to-nearest-front-paw distance, nose height relative to the four-paw ground estimate, nose drop relative to the neck and shoulder midpoint, and the nose-to-neck distance normalised by neck-to-tail trunk length. Nose speed, neck speed, centre speed, paw-height spread and relevant keypoint confidences complete 12 frame-level signals, which contribute \(12\times7=84\) statistics. Twelve duration features measure the proportion and longest continuous run of low-head, strongly-low-head and low-head-with-low-body-motion frames, as well as head-drop percentiles. Six further values describe vertical head-direction changes, nose movement range, accumulated nose/centre motion and valid sequence length. Together with the general summary, this gives \(160+84+12+6=262\) dimensions. These duration features distinguish sustained eating/drinking patterns from a brief downward glance.

**(c) Walking-oriented geometry.** This block adds 22 descriptors to the general summary. They include total centre-path length, net displacement and trajectory straightness,

\[
r_{\mathrm{path}}=
\frac{\lVert\mathbf{c}_{T}-\mathbf{c}_{1}\rVert}
{\sum_{t=2}^{T}\lVert\mathbf{c}_{t}-\mathbf{c}_{t-1}\rVert+\epsilon},
\]

together with speed and acceleration statistics, the proportion of moving frames, direction changes, four-paw speed and vertical-direction reversals, and detection-box scale variation. Walking/running is consequently supported by persistent global translation and coordinated paw dynamics rather than by large joint motion alone.

Thus, the engineered branch combines single-joint trajectories, distances and ratios between selected joints, whole-body detection-box motion, and temporal duration/statistical descriptors. The implementation does not calculate a comprehensive set of anatomical joint angles; “limb-extension ratios and trunk orientation” is a more precise description of the geometry used.

![Engineered geometry and motion cues for standing, eating or drinking, and walking or running](diagrams/report/figure_5_2_engineered_geometry_features.png)

*Figure 5-2. Engineered geometry and motion cues used by the dog-behaviour classifier. The columns show (a) standing, (b) eating/drinking, and (c) walking/running. From top to bottom, each column presents the input image with the estimated AP-10K skeleton, the isolated skeleton with 0-based joint indices, and the corresponding engineered-feature illustration. The blue annotations identify the class-relevant geometry, motion, confidence, duration, trajectory and detection-box signals described in this section.*

#### 5.2.3 Geometry and Pairwise Fusion

For each fold, the TCN probability and the engineered-feature classifier probability are fused in log-probability space:

\[
\mathbf{z}=\log(\mathbf{p}_{\mathrm{TCN}}+\epsilon)
+\alpha\log(\mathbf{p}_{\mathrm{geom}}+\epsilon).
\]

Log-probability fusion allows the geometry branch to adjust the temporal model while retaining the TCN as the principal sequence encoder. The fused logits are converted back to class probabilities before class-pair refinement.

#### 5.2.4 Class-Specialist Models

Two ExtraTrees pairwise specialists refine class pairs that remained difficult in the cross-validation confusion matrix. The first specialist distinguishes standing from eating/drinking. It applies a low-weight correction (\(\alpha=0.1\)) to this pair for every sample, while leaving the other three class logits unchanged. The second specialist distinguishes eating/drinking from sitting and applies a stronger correction (\(\alpha=0.5\)) only when these two classes occupy the top two candidate positions. Each specialist converts its binary probability into a smoothed log-odds value, adds the weighted value to the positive-class logit, and subtracts it from the negative-class logit. This design focuses additional capacity on the two difficult decision boundaries without replacing the original five-class prediction.

The complete dog-behaviour system contains five models produced by five-fold cross-validation. Their output probabilities are averaged during inference. The five stored weights are components of one final ensemble, not five competing behaviour models.

![Five-fold dog-behaviour ensemble with geometry fusion and class-specialist correction](diagrams/report/figure_5_3_dog_behavior_ensemble.png)

*Figure 5-3. Final dog-behaviour ensemble. Within each fold, the Motion-Aware TCN probabilities are fused with the 834-dimensional engineered-geometry classifier in log-probability space. Two ExtraTrees specialists then refine the standing--eating/drinking and eating/drinking--sitting boundaries, with the second correction enabled only when the relevant classes occupy the top two positions. Softmax probabilities from the five independently trained folds are averaged to produce the final five-class prediction.*

### 5.3 Model Three: Human-Dog Interaction Recognition

The interaction model predicts no interaction, hand feeding, petting, or playing together from synchronised COCO-17 human skeletons, AP-10K-17 dog skeletons, detection boxes and selected RGB frames. The design separates interpretable skeleton-relation evidence from complementary visual context.

#### 5.3.1 Human-Dog Relation Features

For each frame, the system extracts a 48-dimensional relation vector from the synchronised human skeleton, dog skeleton and two detection boxes. These features preserve cross-subject geometry that would be lost if the two skeletons were normalised independently. They are organised into five groups:

- **Global person-dog geometry:** relative box-centre displacement, centre distance, relative velocity and closing speed.
- **Hand-to-mouth geometry:** vectors and distances from both COCO wrists to AP-10K joint 2. Because AP-10K has no mouth landmark, the dog's nose is used as a stable mouth proxy. Contact distances are divided by the dog-box diagonal so that their scale remains comparable across near and distant views.
- **Arm-to-trunk contact:** minimum distances from the wrists, elbows, sampled arm segments and human torso to the dog's trunk graph, together with continuous contact scores and binary contact indicators.
- **Posture and contact context:** human and dog orientation, keypoint confidence, detection-box overlap, relative stillness, arm-crossing evidence and an approximate embrace score.
- **Interaction motion:** human and dog box-centre speed, mean and maximum skeleton speed, wrist and elbow speed, acceleration, single-subject activity, joint activity and a playing-motion score.

For a 32-frame clip, every relation channel is summarised by its mean, standard deviation, minimum, maximum, and 25th, 50th and 75th percentiles, producing \(48\times7=336\) values. These order-independent statistics describe the distribution of each relation but cannot represent whether a hand first approaches, remains near and then withdraws from the dog's mouth. The system therefore adds 16 ordered descriptors for hand feeding. They encode the longest near/contact run, stable-contact proportion, start/minimum/end hand-nose distance, approach and withdrawal magnitude, whether the active hand is closer to the nose than to the trunk, dominant-hand consistency, hand-switch frequency, extreme closing and withdrawal speed, and an approach-duration interaction term. The final relation vector has

\[
48\times7+16=352
\]

dimensions.

![Hand-feeding relation changes derived from human and dog skeletons](diagrams/report/figure_5_4_human_dog_relation_features.png)

*Figure 5-4. Hand-feeding example generated from the actual RTMDet and RTMPose outputs for clip `d50_p52_v01_c001`. The upper row shows the human COCO-17 skeleton, dog AP-10K skeleton and their detection boxes at (a) contact onset, (b) the closest hand--nose contact and (c) withdrawal onset. The middle row enlarges the nearest human wrist \(W^{\ast}\) and dog nose \(N_d\). Their scale-normalised distance is \(d_{WN}/\operatorname{diag}(B_d)\), where \(B_d\) is the dog detection box; the three selected frames yield 0.141, 0.108 and 0.185, respectively. The orange arrow denotes wrist displacement direction between the selected frames. The lower plot shows the frame-level distance over the complete clip together with a five-frame smoothed curve. The red dashed line is the implemented contact threshold of 0.25, and the shaded area denotes the contact region. Skeleton links use the official MMPose COCO-17 and AP-10K colour conventions.*

#### 5.3.2 ExtraTrees / Linear-SVM Soft-Vote Ensemble

The relation branch combines an ExtraTrees classifier containing 500 decision trees with a linear SVM. Each tree learns non-linear decisions using relation features such as contact distance, duration and activity, while the linear SVM learns a complementary global boundary on the standardised 352-dimensional vector. Their probabilities are combined by soft voting:

\[
\mathbf{p}_{r}=0.8\mathbf{p}_{\mathrm{ET}}+0.2\mathbf{p}_{\mathrm{SVM}}.
\]

Five classifiers are trained with random seeds 29--33, and their probabilities are averaged to obtain the final relation-branch probability \(\mathbf{p}_{r}\).

![ExtraTrees and linear-SVM soft-vote ensemble for human-dog relation classification](diagrams/report/figure_5_5_relation_soft_vote_ensemble.png)

*Figure 5-5. Relation-branch ensemble for human--dog interaction recognition. The 352-dimensional relation vector is processed by a 500-tree ExtraTrees classifier and a standardised linear SVM. Their class probabilities are combined using weights of 0.8 and 0.2, respectively. The complete classifier is trained with five random seeds, and the resulting probability vectors are averaged to obtain the final four-class relation prediction. The representative tree shows only its first decision levels.*

#### 5.3.3 Frozen CLIP Visual Branch

Skeletons do not explicitly show food, toys or whether physical contact has actually occurred. A frozen CLIP ViT-B/32 encoder is therefore used as an auxiliary visual branch. Its parameters are not fine-tuned, reducing the risk of overfitting on the small interaction dataset.

Eight frames are sampled uniformly from each clip. For every sampled frame, the system generates both a union crop covering the person and dog and a full-scene image, producing 16 images in total. CLIP encodes each image as a 768-dimensional vector. For the crop sequence and full-scene sequence separately, the mean, standard deviation and absolute first-to-last difference are calculated, giving

\[
2\times3\times768=4608
\]

visual features. These features are standardised, reduced to 32 dimensions by PCA and classified by logistic regression. The probabilities of five visual models are averaged to obtain \(\mathbf{p}_{v}\).

#### 5.3.4 Probability-Level Fusion

The relation branch remains the main source of evidence, while visual information is assigned a smaller weight. The final prediction is

\[
\mathbf{p}=\operatorname{softmax}\left[
\log(\mathbf{p}_{r}+\epsilon)+0.1\log(\mathbf{p}_{v}+\epsilon)
\right].
\]

This fusion allows food, toy and contact appearance to refine the relation prediction without allowing the small visual dataset to dominate the decision.

![Dual-branch human-dog interaction recognition and probability fusion pipeline](diagrams/report/figure_5_6_interaction_dual_branch_fusion.png)

*Figure 5-6. Final dual-branch human--dog interaction model. The primary branch summarises synchronised human and dog skeletons and detection boxes into a 352-dimensional relation vector, which is classified by the ExtraTrees/linear-SVM ensemble. The auxiliary branch extracts frozen CLIP ViT-B/32 features from eight person--dog union crops and eight full-scene images, aggregates them into a 4,608-dimensional representation, and applies standardisation, 32-dimensional PCA and logistic regression. The five-model visual probability is combined with the relation probability through weighted log-probability fusion, using weights of 1.0 and 0.1 respectively, before the final four-class Softmax output.*

### 5.4 Alternative Models Explored

LSTM, ST-GCN and Transformer-based skeleton classifiers were explored during development. The LSTM served as a recurrent sequence baseline. ST-GCN explicitly represented joints as graph nodes and physical connections as edges, while the Transformer used attention to model long-range temporal dependencies. These alternatives were valuable for identifying modelling requirements, but the final dog-behaviour system uses the motion-aware TCN and engineered-feature ensemble because it provided the strongest reproducible five-fold result under the current data constraints.

For interaction recognition, skeleton-only temporal and graph models were also considered. However, the clearest interaction cues were explicit cross-subject relations such as hand-to-mouth distance, contact duration and relative activity. The final design therefore uses a relation-feature ensemble, supplemented by frozen visual features. Comparable numerical results for alternative models should only be reported after all methods have been rerun on the same grouped five-fold split.

## 6. System Performance

### 6.1 Experimental Setup

All splits were created before augmentation and feature generation. Clips from the same source video were kept in one partition, and dog and person identities were separated between the development data and held-out test data where possible.

The dog-behaviour dataset contains 170 valid project clips. Of these, 33 clips form a small held-out test set and the remaining 137 clips are used for five-fold grouped cross-validation. A further 31 curated external APT skeleton sequences are used only in the training portion of each fold and never in validation or testing. The 33 test clips contain 6 walking/running, 8 standing, 6 sitting, 7 lying/sleeping, and 6 eating/drinking examples.

For interaction recognition, 75 training clips and 22 validation clips are combined into a 97-clip development set for five-fold grouped OOF evaluation. A separate 24-clip test set contains six examples of each interaction class and has no recorded identity overlap with the development set.

In each cross-validation round, four folds are used for training and one fold is used for validation. After all five rounds, every development clip has exactly one prediction from a model that did not train on that clip. These predictions are concatenated to compute the OOF metrics. Because the held-out test sets are small, OOF Accuracy and OOF Macro-F1 are treated as the primary results; test-set results are reported only as supporting evidence of end-to-end feasibility.

### 6.2 Evaluation Metrics

Accuracy is the proportion of correctly classified clips:

\[
\mathrm{Accuracy}=\frac{N_{\mathrm{correct}}}{N_{\mathrm{all}}}.
\]

Because the class counts are not perfectly balanced, Macro-F1 is also reported. It first calculates the F1 score independently for every class and then gives all classes equal weight:

\[
\mathrm{Macro\text{-}F1}=\frac{1}{C}\sum_{c=1}^{C}\mathrm{F1}_{c}.
\]

Per-class precision, recall and F1, together with confusion matrices, are used for error analysis. Recall is particularly useful for showing which behaviours are missed, but the small number of clips per test class means that each test error changes the percentage substantially.

### 6.3 Dog Pose Estimation Results

The dog pose variants are evaluated on the 36-frame indoor evaluation set (Section 2.4). The primary setting follows the real pipeline: RTMDet detects the dog and the pose model predicts keypoints inside the detected box. Running the pose models inside the hand-drawn boxes is reported as a reference. P1 uses its epoch-19 checkpoint. Both models are exported to ONNX by the same code; re-exporting P0 this way reproduces the official ONNX output exactly, so the two models share identical pre- and post-processing. No flip test is used.

**APT-36K validation.** Fine-tuning improves AP on the APT-36K validation split from 0.775 to 0.817 (Figure 6-1).

![P1 validation AP on APT-36K by epoch](diagrams/report/figure_6_1_p1_apt_val_ap.png)

*Figure 6-1. OKS AP of P1 on the APT-36K dog validation split after each epoch. The dashed line marks P0 before fine-tuning (0.775). AP exceeds P0 from epoch 3, levels off between 0.80 and 0.82 after epoch 10 and peaks at 0.817 at epoch 19, the checkpoint used as P1. Training was stopped after epoch 30 of the 40 planned.*

**Indoor evaluation.** The gain on APT-36K does not carry over to the indoor frames.

| Pose-model input box | Model | OKS AP | AP50 | AP75 | Mean OKS | PCK@0.05 | PCK@0.1 |
|---|---|---:|---:|---:|---:|---:|---:|
| RTMDet box (real pipeline) | P0 | 0.678 | 0.970 | 0.762 | 0.827 | 0.682 | 0.857 |
| RTMDet box (real pipeline) | P1 | 0.693 | 1.000 | 0.754 | 0.831 | 0.680 | 0.861 |
| Hand-drawn box (reference) | P0 | 0.700 | 0.970 | 0.825 | 0.839 | 0.695 | 0.868 |
| Hand-drawn box (reference) | P1 | 0.698 | 1.000 | 0.758 | 0.832 | 0.654 | 0.857 |

Because the same 36 frames are scored by both models, the difference is tested with a paired bootstrap: the per-frame change in OKS (P1 minus P0) is resampled over frames 10,000 times.

| Pose-model input box | Mean OKS change | 95% confidence interval | Improved | Within ±0.01 | Worse |
|---|---:|---|---:|---:|---:|
| RTMDet box | +0.004 | −0.024 to +0.032 | 17 | 7 | 12 |
| Hand-drawn box | −0.007 | −0.028 to +0.013 | 11 | 11 | 14 |

Both intervals include zero, and the sign of the mean change differs between the two box sources. P1 and P0 are therefore tied on indoor frames. Individual frames move in both directions by large amounts, from −0.29 to +0.24 (Figure 6-4), which indicates that fine-tuning changed the model's behaviour on individual dogs without making it better on average. The detection step costs little accuracy: with P0, mean OKS falls from 0.839 with hand-drawn boxes to 0.827 with RTMDet boxes.

**By behaviour.** Lying/sleeping is the weakest posture for both models, with a mean OKS about 0.1 lower than the other labels (Figure 6-2). The remaining differences between P0 and P1 go in both directions and, with six frames per label, are within noise.

| Behaviour label | Frames | P0 mean OKS | P1 mean OKS | Change | P0 PCK@0.1 | P1 PCK@0.1 |
|---|---:|---:|---:|---:|---:|---:|
| All | 36 | 0.827 | 0.831 | +0.004 | 0.857 | 0.861 |
| Walking / Running | 6 | 0.799 | 0.853 | +0.054 | 0.750 | 0.837 |
| Standing | 6 | 0.915 | 0.858 | −0.056 | 0.950 | 0.920 |
| Sitting | 6 | 0.866 | 0.844 | −0.021 | 0.868 | 0.831 |
| Lying / Sleeping | 6 | 0.705 | 0.734 | +0.029 | 0.802 | 0.779 |
| Eating / Drinking | 6 | 0.847 | 0.866 | +0.019 | 0.927 | 0.948 |
| Playing alone | 6 | 0.831 | 0.831 | 0.000 | 0.833 | 0.833 |

![Mean OKS by behaviour label on the indoor evaluation set](diagrams/report/figure_6_2_indoor_oks_by_behaviour.png)

*Figure 6-2. Mean OKS of P0 and P1 for each behaviour label, using RTMDet boxes. Lying/sleeping is the lowest for both models; differences between the models in the other labels are small relative to the six frames per label.*

**By keypoint.** Neither model dominates across keypoints (Figure 6-3). P1 is slightly better on the head, neck and front legs and slightly worse on the tail root, right hip and back legs. Both models are weakest on joints that are often partly hidden by the body: the right shoulder (PCK@0.1 of 0.686 for P0 and 0.771 for P1), the left hip (0.714 for both) and the right hip (0.788 and 0.758). Each keypoint has only 26--36 labelled instances, so a difference of about 0.03 corresponds to a single point.

![PCK@0.1 by keypoint on the indoor evaluation set](diagrams/report/figure_6_3_indoor_pck_by_keypoint.png)

*Figure 6-3. PCK@0.1 of P0 and P1 for each of the 17 AP-10K keypoints, using RTMDet boxes. Grey markers indicate keypoints on which the two models are identical. The shoulders and hips are the weakest joints for both models.*

![Per-frame OKS change from P0 to P1](diagrams/report/figure_6_4_indoor_oks_change_per_image.png)

*Figure 6-4. Per-frame OKS change from P0 to P1 on the 36 indoor frames, using RTMDet boxes and sorted by change. Seventeen frames improve, twelve become worse and seven change by less than 0.01; the largest changes are −0.29 for a standing dog and +0.24 for a lying dog.*

**Comparison with the official benchmark.** The most comparable official figure is the AP-10K large-animal AP, because the official evaluation also uses ground-truth boxes and the indoor dogs occupy a large part of the frame.

| Evaluation data | AP | AP50 | AP75 |
|---|---:|---:|---:|
| Official: AP-10K test set (54 species) | 0.722 | 0.939 | 0.788 |
| Official: AP-10K test set, large animals | 0.728 | -- | -- |
| Indoor, hand-drawn boxes | 0.700 | 0.970 | 0.825 |
| Indoor, RTMDet boxes | 0.678 | 0.970 | 0.762 |

The 95% bootstrap interval of the indoor AP (2,000 resamples over frames) is 0.623--0.775 with hand-drawn boxes and 0.590--0.760 with RTMDet boxes. The official large-animal AP of 0.728 lies inside the first interval, so there is no clear gap between P0's indoor performance and its own benchmark. This comparison is approximate in both directions: the indoor set scores only visible joints, excludes severely blurred frames and contains one large dog per frame, which favours the indoor result, while the official evaluation uses flip testing, which typically adds 0.5--1 AP point and was not used indoors. The official results are also reported for all species together rather than for dogs alone.

**Decision.** P0 is retained as the dog pose model. The APT-36K improvement of about four AP points did not transfer to indoor home-camera scenes, and P0 is already close to its benchmark level. These results have several limitations: the evaluation set has 36 frames and one annotator; occluded joints are not evaluated, although RTMPose still outputs them to the downstream models (with a median confidence of 0.46, compared with 0.79 for labelled joints); frames without a dog, scenes with a real cat and the person boxes were not evaluated; and only one fine-tuning configuration was tried.

### 6.4 Dog Behaviour Recognition Results

The primary five-fold OOF result of the final dog-behaviour ensemble is **77.37% Accuracy** and **76.47% Macro-F1**. This is the main quantitative estimate reported for model comparison. On the additional 33-clip test set, the model obtains 78.79% Accuracy and 78.83% Macro-F1. The test result supports the feasibility of the approach but is not used as the principal performance claim because of the limited sample count.

| Evaluation | Clips | Accuracy | Macro-F1 | Role in the report |
|---|---:|---:|---:|---|
| Five-fold OOF | 137 project clips | **77.37%** | **76.47%** | Primary result |
| Small held-out test | 33 | 78.79% | 78.83% | Feasibility evidence |

![Five-fold OOF confusion matrix for dog-behaviour recognition](diagrams/report/figure_6_1_dog_behavior_oof_confusion_matrix.png)

*Figure 6-5. Five-fold out-of-fold confusion matrix for the final dog-behaviour model. Each cell reports the percentage within its actual class. The model is strongest on sitting, lying/sleeping and walking/running, while standing and eating/drinking remain the principal sources of confusion.*

The test-set recalls are shown below for supplementary class-level analysis.

| Dog behaviour class | Test clips | Correct | Recall |
|---|---:|---:|---:|
| Walking / Running | 6 | 5 | 83.33% |
| Standing | 8 | 7 | 87.50% |
| Sitting | 6 | 5 | 83.33% |
| Lying / Sleeping | 7 | 6 | 85.71% |
| Eating / Drinking | 6 | 3 | 50.00% |

### 6.5 Human-Dog Interaction Recognition Results

The interaction experiments compare the skeleton-relation branch, the frozen CLIP visual branch and their final fusion.

| Method | OOF Accuracy (primary) | OOF Macro-F1 (primary) | Small-test Accuracy (supporting) | Small-test Macro-F1 (supporting) |
|---|---:|---:|---:|---:|
| Relation branch | 77.32% | 75.89% | 70.83% | 69.36% |
| CLIP visual branch | 60.82% | 60.78% | 75.00% | 73.40% |
| Relation + CLIP (α = 0.1) | **77.32%** | **76.13%** | **83.33%** | **83.30%** |

The relation branch is substantially stronger than the visual branch in five-fold OOF evaluation and is therefore the principal component of the interaction model. Adding CLIP leaves OOF Accuracy unchanged and increases OOF Macro-F1 slightly from 75.89% to 76.13%. The higher result on the 24-clip test set indicates that appearance may help with food, toys and physical-contact cues; because the test set is small, this result is interpreted as evidence that the fusion scheme is feasible rather than as the main generalisation estimate.

![Five-fold OOF confusion matrix for human-dog interaction recognition](diagrams/report/figure_6_2_interaction_oof_confusion_matrix.png)

*Figure 6-6. Five-fold out-of-fold confusion matrix for the final relation-plus-CLIP interaction model. Each cell reports the row-normalised percentage. No interaction and playing together have the highest recall; hand feeding is most often confused with petting, reflecting their shared close hand-to-dog geometry.*

The following test recalls are included only as supplementary observations.

| Interaction class | Test clips | Correct | Recall |
|---|---:|---:|---:|
| No Interaction | 6 | 5 | 83.33% |
| Hand Feeding | 6 | 5 | 83.33% |
| Petting | 6 | 4 | 66.67% |
| Playing Together | 6 | 6 | 100.00% |

### 6.6 Ablation Studies

The dog-behaviour ablation follows the construction of the final system incrementally. All three variants use the same 137 OOF clips and the same grouped five-fold split; the 31 curated external APT sequences are available only to the training portion of each fold. The base Motion-Aware TCN already contains joint velocity, global detection-box displacement and confidence-weighted temporal pooling. The next stage adds the 834-dimensional clip-level geometry classifier, and the final stage adds the two conditional pairwise corrections.

| Dog-behaviour variant | Component added at this stage | OOF Accuracy | OOF Macro-F1 | Change from previous stage |
|---|---|---:|---:|---:|
| Motion-Aware TCN | Temporal skeleton and global-motion backbone | 70.80% | 68.88% | — |
| TCN + 834-D geometry | Engineered clip-level geometry fusion | 75.91% | 74.12% | +5.11 pp Acc.; +5.23 pp F1 |
| TCN + geometry + pairwise correction | Two conditional class specialists | **77.37%** | **76.47%** | +1.46 pp Acc.; +2.35 pp F1 |

The 834-dimensional geometry branch provides the largest gain, increasing Accuracy by 5.11 percentage points and Macro-F1 by 5.23 points over the TCN backbone. The pairwise correction stage provides a further improvement of 1.46 Accuracy points and 2.35 Macro-F1 points. This supports the use of explicit geometry and targeted correction under the current small-data setting.

For the interaction model, the currently available relation-only, visual-only and fused results already provide a branch-level component comparison. Further experiments can separately remove the linear SVM, the union crop and the full-scene CLIP features to identify which parts of the fusion are beneficial.

The interaction ablation follows the construction of the final model step by step. ExtraTrees is first evaluated as the non-linear relation baseline. A linear SVM branch is then added through 0.8/0.2 soft voting, after which the auxiliary CLIP branch is introduced with a visual log-probability weight of 0.1. All three variants use exactly the same five grouped folds. The final column reports both Accuracy and Macro-F1 across the five folds.

| Interaction variant | Purpose | OOF Accuracy | OOF Macro-F1 | Fold mean ± standard deviation |
|---|---|---:|---:|---|
| ExtraTrees only | Non-linear tree-ensemble baseline | 74.23% | 72.98% | Acc. 74.37% ± 8.20%; F1 73.18% ± 8.44% |
| ExtraTrees + Linear SVM | Contribution of the linear branch and heterogeneous soft voting | **77.32%** | 75.89% | Acc. 77.42% ± 5.78%; F1 75.83% ± 6.62% |
| Relation ensemble + CLIP | Contribution of the auxiliary visual branch | **77.32%** | **76.13%** | Acc. 77.42% ± 6.67%; F1 76.41% ± 6.86% |

The aggregated OOF metrics measure overall predictive performance, while the mean and standard deviation across folds indicate stability across data subsets. Adding the linear SVM branch to ExtraTrees raises OOF Accuracy from 74.23% to 77.32% and Macro-F1 from 72.98% to 75.89%; it also reduces the fold-level Accuracy standard deviation from 8.20% to 5.78%. Adding CLIP leaves Accuracy unchanged and further raises Macro-F1 to 76.13%. The complete model therefore achieves the highest Macro-F1 in this incremental ablation, while matching the relation ensemble in Accuracy. A formal bias--variance decomposition is unnecessary here; the controlled ablation and fold dispersion provide more directly interpretable evidence.

### 6.7 Runtime and Deployment Considerations

## 7. Findings and Discussion

### 7.1 Key Findings

The five-fold results support three main findings. First, preserving both local joint motion and global detection-box displacement is important for dog behaviour recognition. Second, explicit human-dog relation features are more reliable than visual appearance alone for the current interaction dataset. Third, frozen CLIP features can provide complementary object and contact context, but their contribution is smaller and should remain auxiliary under the current data scale.

### 7.2 Error Analysis

The main dog-behaviour errors occur between eating/drinking and standing or lying/sleeping. In several clips, the dog keeps its head low while the mouth, food or water container is occluded. The skeleton then describes head and neck posture but cannot confirm whether ingestion occurs. Future versions should add mouth-region visual information and difficult negative clips in which a dog lowers its head to sniff without eating.

For interaction recognition, petting and hand feeding both contain a hand approaching the dog's head or trunk, while playing together can also contain brief physical contact. Contact duration and the approach--hold--withdraw pattern reduce this ambiguity, but the present dataset does not cover enough variations in stroking, embracing, feeding style and viewpoint.

### 7.3 Impact of Detection and Tracking Quality

Both recognition tasks depend on the quality of upstream detection and pose estimation. Missing paws can change support-pattern features, inaccurate head keypoints can affect eating cues, and unstable wrist or nose positions can strongly alter interaction distances. Confidence-weighted pooling, temporal interpolation and sequence smoothing reduce short errors, but long occlusions cannot be reconstructed reliably. In the current one-person, one-dog scope, identity tracking is not required for resolving multiple subjects; nevertheless, consistent detections across the 32-frame window remain necessary.

### 7.4 Data Quality, Size and Leakage

The dataset was organised by source video and subject identity rather than by randomly splitting individual frames. This prevents near-duplicate clips from the same video from appearing across training and evaluation partitions. External APT skeleton sequences are restricted to the training folds. Despite these controls, the number of identities and clips remains limited, particularly for subtle or visually dependent categories. Consequently, five-fold OOF metrics are treated as the principal results, while the small test sets are used only to demonstrate feasibility.

### 7.5 Limitations

The current system assumes one principal person and one principal dog and does not resolve identity association in multi-person or multi-dog scenes. The pose representation is two-dimensional and is sensitive to viewpoint, self-occlusion and truncation. The visual branch uses a frozen general-purpose CLIP encoder rather than a model adapted specifically to indoor human-dog interaction. Finally, the predicted labels describe observable actions only and should not be interpreted as diagnoses of animal health, emotion or safety.

### 7.6 Future Work

Future work should collect more identity-independent videos for the weak and confusable classes, especially eating/drinking, petting and hand feeding. A larger locked evaluation set would provide a more stable final estimate. Model development could also investigate mouth-centred visual features, more explicit contact-duration modelling, three-dimensional pose cues, and multi-subject association. Once sufficient data are available, the TCN, ST-GCN and Transformer alternatives should be rerun under the same grouped protocol for a complete controlled comparison.

## 8. Conclusion

## References

1. Bai, S., Kolter, J. Z., and Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. arXiv:1803.01271.
2. Geurts, P., Ernst, D., and Wehenkel, L. (2006). “Extremely Randomized Trees.” *Machine Learning*, 63, 3--42.
3. Radford, A., et al. (2021). “Learning Transferable Visual Models From Natural Language Supervision.” *Proceedings of the 38th International Conference on Machine Learning*.
4. Yan, S., Xiong, Y., and Lin, D. (2018). “Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition.” *Proceedings of the AAAI Conference on Artificial Intelligence*.

## Appendix

### A. Installation and User Guide

### B. Model Training and Reproduction Commands

### C. Dataset and Model File Listing

### D. Additional Results
