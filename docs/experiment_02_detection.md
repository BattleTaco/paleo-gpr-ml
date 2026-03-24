# Experiment 02: GPR B-scan Object Detection

## Objective

Move from "which class is this image?" to "where is the anomaly?" Using the YOLO bounding box annotations on cavities and utilities, train an object detection model to localize subsurface anomalies in GPR B-scans.

This is a harder task than classification. The model needs to both identify that an anomaly exists and predict its spatial location. This is the stepping stone to fossil-like anomaly detection.

## Why detection matters more than classification for our research

Classification tells us "this B-scan contains a cavity." Detection tells us "there's a cavity centered at (x=112, y=156) with approximate extent 60x45 pixels." For paleontological prospecting, we need the second thing. A field archaeologist needs to know *where to dig*, not just that something is there.

## Dataset

- **Positive samples**: Augmented cavities (553 images, 1 bbox each) + augmented utilities (786 images, 1-3 bboxes each)
- **Negative samples**: Augmented intact (900 images, no objects) used as background/negative class
- **Annotations**: YOLO format already available (class_id, cx, cy, w, h, all normalized)
- **Image format**: 224x224 grayscale JPEG
- **Split**: Same by-original-ID split as Experiment 01

## Detection framing

Two options, and we should try both:

### Option A: 2-class detection
- Class 0: cavity
- Class 1: utility
- Intact images as negative samples (no detections expected)
- This is the natural framing given our annotations

### Option B: Binary anomaly detection
- Class 0: anomaly (both cavities and utilities)
- Intact as negative
- Simpler task, might generalize better to fossil-like targets (since fossils are "anomaly, type unknown")

## Models to compare

### YOLOv8 (primary)
- Ultralytics YOLOv8n or YOLOv8s (nano or small)
- Why: fast, well-supported, strong on small datasets with augmentation, single-stage detector
- Input: 224x224 or upscale to 640x640 (YOLO's native size)
- Pretrained on COCO, fine-tune on our data

### Faster R-CNN (secondary)
- torchvision Faster R-CNN with ResNet50-FPN backbone
- Why: two-stage detector, different inductive biases than YOLO. The GPR AI review literature shows both YOLO and R-CNN family used for GPR detection.
- Pretrained on COCO, fine-tune

### ResNet18 as detector (control)
- Take our trained ResNet18 classifier
- Use Grad-CAM heatmap thresholding as a pseudo-detection method
- This gives us a "free" detection baseline from the classification model

## Training

- Optimizer: SGD with momentum (YOLO default) or AdamW
- LR: follow model defaults, with cosine scheduling
- Epochs: 100-200 with early stopping on val mAP
- Augmentation: mosaic (YOLO default), horizontal flip, brightness/contrast jitter
- Class weights or focal loss to handle imbalance

## Evaluation metrics

- **mAP@0.5** (mean average precision at IoU threshold 0.5): primary metric
- **mAP@0.5:0.95**: stricter, averages across IoU thresholds
- **Per-class AP**: how well does each class do?
- **Precision-Recall curves**: per class
- **False positive rate on intact images**: how often does the model hallucinate detections on clean B-scans?
- **Detection visualization**: predicted boxes overlaid on images alongside ground truth

## Success criteria

- **Minimum**: mAP@0.5 > 0.50 for 2-class detection. This confirms anomalies are localizable.
- **Good**: mAP@0.5 > 0.70 with reasonable precision on intact images (< 10% false positive rate).
- **Excellent**: mAP@0.5 > 0.85. Would suggest the model can reliably localize GPR anomalies.

## Known risks

1. **Small objects**: Some bounding boxes are small relative to 224x224. Detection models struggle with small objects.
2. **Single annotation per cavity image**: Only 1 box per cavity image. If there are multiple visible anomalies, the missing annotations hurt training.
3. **Spatial bias**: Bbox positions cluster by class (EDA showed this). The model might learn "predict a box at y=150 for utilities" rather than learning the actual pattern.
4. **Resolution**: 224x224 is small for detection. Upscaling to 640x640 adds computation but gives the model more spatial resolution.

## Deliverables

- Trained YOLO and R-CNN checkpoints
- mAP tables and PR curves
- Detection visualization grid (predicted vs. ground truth)
- False positive analysis on intact images
- Comparison table: YOLO vs. R-CNN vs. Grad-CAM pseudo-detection
- Notes in `docs/notes/05_detection_experiment_notes.md`
- Results logged in `docs/research_log.md`

## Dependencies

- Run Grad-CAM analysis first (Experiment 01b / notebook 03) to understand what the classification model sees
- Ultralytics package for YOLOv8
- Data reformatted into YOLO detection directory structure (images/ and labels/ subdirs)

## Files to create

- `src/data/prepare_detection_data.py`: Reformat data for YOLO and R-CNN training
- `configs/detection_yolov8.yaml`: YOLO training config
- `notebooks/04_object_detection.ipynb`: Main detection experiment notebook
- `docs/notes/05_detection_experiment_notes.md`: Analysis notes
