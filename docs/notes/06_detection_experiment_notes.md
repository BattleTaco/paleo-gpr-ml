# Object Detection Experiment Notes

**Date**: 2026-03-23
**Notebook**: `notebooks/04_object_detection.ipynb`
**Experiment spec**: `docs/experiment_02_detection.md`

---

## Results Summary

| Model | mAP@0.5 | mAP@0.5:0.95 | FP Rate (intact) |
|---|---|---|---|
| YOLOv8n (2-class: cavity vs utility) | 0.581 | 0.296 | 0.7% (1/148) |
| YOLOv8n (binary: anomaly) | 0.600 | 0.312 | -- |

Per-class AP@0.5 for 2-class model:
- Cavity: 0.538
- Utility: 0.625

## Interpretation

**Detection works, and it's a harder task than classification.** mAP@0.5 around 0.58-0.60 is solidly above the 0.50 minimum bar. Classification was 99.5% accuracy, but detection requires both finding the right region and having sufficient box overlap. The gap between classification and detection confirms what the Grad-CAM analysis predicted: the model can tell classes apart easily, but localizing the exact anomaly region is harder.

**Binary is slightly better than 2-class.** 0.600 vs 0.581 mAP@0.5. Not a huge difference, but collapsing to a single anomaly class removes the need to distinguish cavity from utility during detection. For our eventual goal of fossil detection (where we just need "anomaly, investigate further"), binary is the more relevant framing.

**Utilities are easier to detect than cavities.** 0.625 vs 0.538 AP@0.5. This is interesting because the Grad-CAM analysis showed the classification model *doesn't* focus on utility bboxes. The explanation: YOLO is a detection model that explicitly learns to localize, while the classification model doesn't need to. The hyperbolic signatures of utilities are geometrically distinctive and localizable even though the classification model was using broader context.

**False positive rate is extremely low.** Only 1 out of 148 intact test images triggered a false detection (0.7%). This means the detector reliably distinguishes "something is there" from "nothing here." This is exactly what we need for a fossil prospecting tool.

**mAP@0.5:0.95 is lower (~0.30).** This stricter metric averages across IoU thresholds from 0.5 to 0.95. The relatively low score means the predicted boxes don't tightly align with ground truth. This could be because: (a) the annotated boxes are approximate, (b) GPR anomaly boundaries are inherently fuzzy, or (c) the model is getting the rough location right but the exact extent wrong. For prospecting, approximate localization is fine.

## What this means for the project

1. **Detection baseline is established.** mAP@0.5 ~0.60 with very low false positives. This is a working GPR anomaly detector.

2. **Binary framing is the way to go for fossil work.** The binary model performs as well or better, and it's the right abstraction for "is there something interesting at this location?"

3. **Ready for synthetic data experiment.** We now have: (a) a classification model that works, (b) a detection model that works, (c) Grad-CAM analysis showing what the model learns, (d) a synthetic data plan with physics-based parameters. The next step is to generate synthetic fossil-like GPR data with gprMax and test whether the detector can find those targets.

4. **The mAP numbers are honest.** With by-ID splits and a hard detection task, ~0.60 mAP is a reasonable baseline. Not inflated, not trivial.

## Lessons

- Detection is genuinely harder than classification on this dataset. The 99.5% classification accuracy was misleading about the difficulty of the actual task.
- Binary anomaly detection is marginally better than 2-class, probably because it doesn't waste capacity on distinguishing target types.
- The near-zero false positive rate on intact images is a strong result. The model confidently separates "something" from "nothing."
- YOLOv8n (nano, 3M params) is sufficient for this dataset. A larger model might improve mAP slightly but probably not dramatically.
