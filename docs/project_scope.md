# Project Scope: paleo-gpr-ml

## 1. Vision

I am building a research-oriented machine learning system that explores **non-invasive fossil prospecting using Ground-Penetrating Radar (GPR)** and extends into **fossil identification using computer vision**.

The long-term vision is a **two-stage pipeline**:
1) Detect fossil-like anomalies underground using GPR
2) Analyze recovered fossils using image/CT-based ML

This project bridges:
- Geophysics (GPR)
- Computer Vision
- Paleontology

---

## 2. Version 1 Scope (CRITICAL)

### Objective
Build a **working ML prototype** that detects fossil-like anomalies in GPR radargrams using:
- Open GPR datasets
- Synthetic data generation

### Explicitly INCLUDED
- GPR data ingestion + preprocessing
- Radargram visualization and analysis
- Synthetic fossil-like data generation
- Baseline ML model for anomaly detection
- Evaluation metrics and error analysis

### Explicitly EXCLUDED (for now)
- Real dinosaur field validation
- Full geological modeling
- End-to-end excavation system
- Robotics or hardware integration
- Complex 3D multi-scan fusion

---

## 3. Core Research Question

Can machine learning detect **fossil-like subsurface anomalies** in GPR radar data using a combination of:
- synthetic data
- real-world non-fossil GPR data

---

## 4. Hypothesis

A model trained on:
- synthetic bone-like structures
- real GPR anomaly datasets

can generalize to detect **previously unseen fossil-like targets**.

---

## 5. Pipeline Architecture

### Stage A: Subsurface Detection (V1 focus)

Input:
- GPR radargrams (2D B-scans)

Processing:
- normalization
- denoising
- augmentation

Model:
- YOLO (object detection) OR
- segmentation model (U-Net / Mask R-CNN)

Output:
- bounding boxes OR masks of anomalies

---

### Stage B: Fossil Analysis (V2)

Input:
- images or CT scans of extracted fossils

Model:
- classification (CNN / ViT)
- segmentation (U-Net / DeepLab)

Output:
- fossil type
- morphology segmentation

---

## 6. Success Criteria

### Minimum Success (V1)
- Successfully train a model on GPR data
- Detect anomalies with measurable precision/recall
- Demonstrate ability to detect synthetic fossil-like targets

### Strong Success
- Model generalizes across datasets
- Clear improvement over baseline
- Meaningful error analysis

### Exceptional Success
- Synthetic → real transfer works well
- Publishable insights
- Dataset or benchmark contribution

---

## 7. Risks and Challenges

- Lack of real fossil GPR data
- Domain gap between synthetic and real data
- Noisy radar signals
- Ambiguity between rocks vs fossils

---

## 8. Key Assumptions

- Fossils produce distinguishable radar signatures
- Synthetic simulation can approximate reality
- ML models can learn anomaly patterns

---

## 9. Deliverables

- Clean GitHub repo
- Literature review
- Dataset pipeline
- Baseline model
- Results + visualizations
- Technical report or paper draft

---

## 10. Future Extensions

- 3D GPR modeling
- multi-modal fusion (GPR + CT)
- field collaboration
- real fossil validation