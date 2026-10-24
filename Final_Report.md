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

### 2.2 Dog Behaviour Clips

### 2.3 Human-Dog Interaction Clips

### 2.4 Dog Pose Data (Hand-Labelled Evaluation Set and APT-36K)

### 2.5 Annotation and Data Preparation Pipeline

### 2.6 Data Splits and Leakage Control

## 3. Tools and Techniques

### 3.1 Software Stack and Frameworks

### 3.2 Object Detection (RTMDet)

### 3.3 Pose Estimation (RTMPose)

### 3.4 Multi-Target Tracking and Skeleton Extraction

### 3.5 Skeleton Preprocessing and Normalisation

### 3.6 Temporal Modelling (TCN)

### 3.7 Feature Engineering and Classical Classifiers

### 3.8 Ensemble Learning

### 3.9 Evaluation Methods

## 4. System Design

### 4.1 Overall System Architecture

### 4.2 Dual-Species Skeleton Extraction Pipeline

### 4.3 Repository and Module Structure

### 4.4 Frontend Prototype

## 5. Models

### 5.1 Model One: Dog Pose Estimation

#### 5.1.1 Baseline (P0: RTMPose-m AP-10K)

#### 5.1.2 Fine-Tuning on APT-36K (P1)

#### 5.1.3 Further Adaptation (P2)

### 5.2 Model Two: Dog Behaviour Recognition

#### 5.2.1 Motion-Aware TCN Backbone

#### 5.2.2 Engineered Geometry Features

#### 5.2.3 Geometry and Pairwise Fusion

#### 5.2.4 Class-Specialist Models

### 5.3 Model Three: Human-Dog Interaction Recognition

#### 5.3.1 Human-Dog Relation Features

#### 5.3.2 ExtraTrees / Linear-SVM Soft-Vote Ensemble

### 5.4 Alternative Models Explored

## 6. System Performance

### 6.1 Experimental Setup

### 6.2 Evaluation Metrics

### 6.3 Dog Pose Estimation Results

### 6.4 Dog Behaviour Recognition Results

### 6.5 Human-Dog Interaction Recognition Results

### 6.6 Ablation Studies

### 6.7 Comparison with Alternative Models

### 6.8 Runtime and Deployment Considerations

## 7. Findings and Discussion

### 7.1 Key Findings

### 7.2 Error Analysis

### 7.3 Impact of Detection and Tracking Quality

### 7.4 Data Quality, Size and Leakage

### 7.5 Limitations

### 7.6 Future Work

## 8. Conclusion

## References

## Appendix

### A. Installation and User Guide

### B. Model Training and Reproduction Commands

### C. Dataset and Model File Listing

### D. Additional Results
