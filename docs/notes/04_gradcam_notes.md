# Grad-CAM Analysis Notes

**Date**: 2026-03-23
**Notebook**: `notebooks/03_gradcam_analysis.ipynb`
**Models analyzed**: ResNet18 (99.5% test acc), SimpleCNN (96.7% test acc)

---

## Purpose

Verify that the high classification accuracy comes from learning real GPR features rather than dataset artifacts. The models might be exploiting:
- Border/edge patterns specific to the augmentation pipeline
- Spatial position bias (cavities tend to be in certain image regions)
- Surface wave intensity (top few rows) rather than subsurface anomalies
- Augmentation-specific contrast or texture artifacts

Grad-CAM reveals where in the image the model focuses when making each prediction.

## What to look for when running the notebook

### Per-class Grad-CAM grids (Section 3)
- **Cavities**: Hot regions should overlap with the bright subsurface patches. If attention is on image edges or uniformly spread, that's suspicious.
- **Utilities**: Hot regions should overlap with the hyperbolic reflections, especially near the apex of each hyperbola. This is the most diagnostic pattern.
- **Intact**: Attention should be diffuse or focused on the smooth gradient. There's nothing specific to find, so a spread-out heatmap is expected and fine.

### Grad-CAM vs. Bounding Box overlap (Sections 4-5)
- High `attention_in_box` means the model's focus is correctly placed inside the anomaly annotation
- High `box_covered` means the model sees the whole anomaly, not just a corner of it
- If both metrics are high (>0.5), the model is genuinely learning anomaly patterns
- If `attention_in_box` is low, the model might be using context outside the anomaly (background texture) to classify

### Average attention maps (Section 6)
- Check for spatial bias. If the average attention map shows a fixed hotspot regardless of where anomalies actually are, the model learned a spatial shortcut.
- Compare the average attention maps between classes. They should look different because the anomaly signatures are in different spatial distributions.

### Depth profiles (Section 7)
- Utilities should peak at mid-to-lower depths (where hyperbola apexes are)
- Cavities should peak at mid-depth
- Intact should be relatively flat
- If all three classes peak at the same depth (e.g., top 20 pixels = surface wave), that's a bad sign

## Results

### Quantitative overlap (Section 5)

| Class | Attention in BBox (mean) | Attention in BBox (median) | BBox Covered (mean) | BBox Covered (median) |
|---|---|---|---|---|
| Cavities | 0.513 | 0.491 | 0.881 | 0.917 |
| Utilities | 0.185 | 0.120 | 0.276 | 0.231 |

### Findings

**Cavities: the model genuinely attends to the anomaly region.** Attention-in-bbox of 0.51 and bbox-covered of 0.88 means the model's focus overlaps substantially with the annotated cavity locations. The high bbox coverage (88%) tells us the model sees most of the anomaly, not just a sliver. About half of the high-attention area falls inside the bbox, with the rest likely on surrounding context (which could be informative for classification anyway).

**Utilities: the model looks elsewhere.** Attention-in-bbox of only 0.185 means the model's focus is mostly *not* on the annotated utility bounding boxes. Bbox coverage is low too (28%). This is surprising for a model with 99.5% accuracy. The model is correctly classifying utilities but appears to use features *outside* the annotated region -- possibly the broader hyperbolic tails, the overall texture pattern, or the background characteristics that differ between utility and non-utility images.

**This is not necessarily bad, but it's worth understanding.** For classification, using context beyond the bounding box can be effective. But for detection (where we need to localize), a model that doesn't focus on the actual target is concerning. It means the classification model's features may not directly transfer to detection.

### Interpretation

The divergence between cavities and utilities makes physical sense:

- **Cavity signatures are localized.** A subsurface void produces a relatively compact bright patch. The model can see it by looking at that region.
- **Utility signatures are extended.** A buried pipe produces a hyperbola whose tails extend far beyond the annotated bbox (which only covers the apex region). The model may be using the full hyperbolic structure, the background texture change caused by the pipe, or the overall edge density of the image.

The utility result is consistent with the model doing "whole-image texture classification" rather than "find the object." This works for classification but won't directly translate to detection.

### Implications for detection experiment

1. **Detection models need to learn localization explicitly.** We can't just threshold Grad-CAM to get detections, at least not for utilities. A real detection model (YOLO, Faster R-CNN) with box regression is necessary.

2. **Cavity detection should be easier than utility detection.** The classification model already attends to the right region for cavities. For utilities, the detection model needs to learn to focus on the hyperbola apex, not just the diffuse pattern.

3. **Binary anomaly framing might help utilities.** If the model can learn "something is here" from the whole-image pattern and then refine to a bbox, that's a two-stage process. Alternatively, training detection directly may force the model to learn localized features.

4. **Grad-CAM as pseudo-detection baseline.** For cavities, thresholding Grad-CAM at 0.5 gives decent localization (51% attention in bbox, 88% coverage). For utilities, it would be a very weak detector. Worth quantifying as a baseline but not viable as a method.

---

## Connection to literature

The Vertebrate Skeleton Detection paper (Peredo et al.) found that bone signatures appear at specific depth ranges determined by the burial depth and permittivity contrast. If our model's attention depth profile matches the known anomaly depth distribution in our dataset, that's consistent with the model learning real physics.

The GPR AI Review (Kücükdemirci & Sarris) noted that most GPR ML studies don't validate *what* the model learns. This Grad-CAM analysis puts us ahead of most published work in terms of interpretability.
