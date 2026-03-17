# Project Ideas Expansion

## Core Idea
Use ML to detect fossil-like structures underground using GPR and classify them post-extraction.

---

## Idea 1: Synthetic Fossil Benchmark (HIGH PRIORITY)

### Description
Build a dataset of synthetic GPR scans with fossil-like objects.

### Why it matters
- No public fossil GPR datasets
- Huge research gap
- Publishable dataset

### Tasks
- use gprMax
- simulate:
  - bones
  - ribs
  - clusters
- generate labels

---

## Idea 2: GPR Anomaly Detector

### Description
Train a model to detect underground anomalies.

### Models
- YOLO
- Mask R-CNN
- U-Net

### Output
- bounding boxes
- anomaly maps

---

## Idea 3: Fossil Image Classifier

### Dataset
- Fossil Image Dataset (FID)

### Task
Classify fossil type

### Models
- EfficientNet
- ResNet
- ViT (later)

---

## Idea 4: Fossil CT Segmentation

### Dataset
- MorphoSource / ESRF

### Task
Segment fossil from rock

### Models
- U-Net
- 3D U-Net
- DeepLab v3+

---

## Idea 5: Multimodal Pipeline (ENDGOAL)

Combine:
- GPR detection
- Fossil classification

---

## Idea 6: Transfer Learning Study

### Question
Does training on pipes/voids transfer to fossils?

---

## Idea 7: Explainability

Use:
- Grad-CAM
- saliency maps

---

## Recommended Starting Idea

Start with:
1. GPR anomaly detection
2. Synthetic data generation

This is your strongest research contribution path.