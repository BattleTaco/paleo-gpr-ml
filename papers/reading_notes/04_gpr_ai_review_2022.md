# Reading Notes: GPR Data Processing and Interpretation Based on AI (2022)

**Citation**: Kücükdemirci, M. & Sarris, A. (2022). GPR Data Processing and Interpretation Based on Artificial Intelligence Approaches: Future Perspectives for Archaeological Prospection. *Remote Sensing*, 14, 3377.
**Read**: 2026-03-23

---

## What this paper is about

Review of ML/DL methods applied to GPR data, with a focus on archaeological prospection. Covers both B-scan and C-scan (time-slice) analysis. Catalogs the state of the art as of 2022 for automated GPR interpretation.

## Key findings

**GPR data comes in four formats, and each has different ML approaches:**
1. A-scans (individual traces) - mostly used for inversion and classification of single traces
2. B-scans (2D radargrams) - the format our dataset uses. Object detection and classification.
3. C-scans (time/depth slices) - horizontal slices showing spatial distribution of reflectors. Segmentation is the natural task.
4. 3D volumes - least explored with ML. Future direction.

**What's been done on B-scans:**
- SVM classifiers for recognizing geometric shapes (cubes, cylinders, spheres) from synthetic data [Ali et al.]
- Cascade R-CNN for object detection [Chen et al.]
- Faster R-CNN for classification and recognition [Gong et al.]
- LeNet for feature extraction [Elsaadany et al.]
- Faster R-CNN for detecting buried objects [Pham et al.]
- Most of these used lab-controlled data or synthetic data from gprMax

**Critical finding: most B-scan studies used synthetic data.** Because labeled real GPR data is scarce, researchers commonly generate synthetic training data with gprMax. This directly validates our planned synthetic data pipeline.

**Archaeological applications specifically:**
- Verdonck: R-CNN for detecting diffraction hyperbolas in B-scans
- Green & Cheetham: ML classification for detecting graves (trained on ~1000 images, enriched with GprSIM and gprMax synthetic data, used Inception V3/VGG/ResNet, best was ResNet152 at 94% accuracy)
- Kücükdemirci & Sarris: U-shaped CNN for segmenting archaeological features in time slices, trained from scratch on ~2000 annotated images with augmentation to 4000, achieved 92% Dice
- Manataki et al.: AlexNet for C-scan classification into three categories (geological, archaeological, noise), 92% accuracy

**The transfer learning vs. training from scratch question:**
- Green's work used ImageNet-pretrained models and got 94% on grave detection
- Kücükdemirci & Sarris trained from scratch on domain-specific GPR data and got 92% Dice for segmentation
- Both approaches work. Transfer learning is more sample-efficient, training from scratch gives better domain alignment.

**Limitations flagged by the authors:**
- Scarcity of annotated data is the #1 bottleneck
- Simulated data (gprMax) is useful for hyperparameter tuning but not a replacement for real data
- Quality of annotations matters as much as quantity
- Data augmentation (shear, zoom, flip, rotation, crop) is effective for increasing training set
- Overfitting and generalization are the main risks with small datasets

## What I want to steal from this paper

- **The precedent of synthetic + augmented GPR data for training.** Green et al. used GprSIM + gprMax synthetic data combined with real data. We should do the same.
- **ResNet152 as a strong baseline for GPR classification.** Our ResNet18 already crushes it, but for harder tasks (detection, cross-domain), a deeper backbone might be needed.
- **The U-shaped architecture for segmentation.** If we move to segmentation instead of bounding box detection, U-Net is the established architecture for GPR.
- **The explicit note that simulated data is better for tuning than for training the final model.** This means our synthetic data strategy should include a fine-tuning step on real data.
- **R-CNN family for B-scan object detection.** Faster R-CNN has been used successfully for GPR object detection. We should compare this with YOLO.

## Limitations of this paper

- Published 2022, so misses the recent surge in transformer-based detection and YOLO v8+
- Limited to archaeological applications. Doesn't cover infrastructure (our dataset domain) or paleontology.
- Thin on implementation details. It's a high-level survey.
- Doesn't cover domain adaptation or transfer between different GPR domains.

## Relevance to our project: HIGH

Maps the landscape of what's been done with ML + GPR. Key takeaways: synthetic data is standard practice, R-CNN and YOLO are the detection architectures to try, and the annotation bottleneck is real. Our approach of starting with infrastructure data and bridging to fossil detection through synthetic data fits within the established methodology but targets a new application.

## 3-2-1

**3 things I learned:**
1. Multiple groups have used gprMax synthetic data to supplement small real GPR datasets for training. It's an accepted approach, not a hack.
2. For B-scan object detection, Faster R-CNN and YOLO are both in active use. Green's grave detection work at 94% accuracy with ResNet152 is a good benchmark.
3. The authors explicitly state that simulated data should be used for hyperparameter tuning, not as a substitute for real data in the final model.

**2 things I still don't know:**
1. What the performance gap typically looks like between models trained on synthetic-only vs. synthetic+real GPR data.
2. Whether any of the cited methods have been tested across different GPR equipment or survey sites (cross-domain generalization).

**1 thing I want to investigate:**
Try Faster R-CNN in addition to YOLO for our detection task on the cavity/utility annotations. The literature suggests both work, and comparing them on our specific data would be informative.
