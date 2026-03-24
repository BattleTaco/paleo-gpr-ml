# GPR Dataset EDA Notes

**Date**: 2026-03-23
**Notebook**: `notebooks/01_gpr_data_exploration.ipynb`

---

## What I Set Out To Do

Before making any modeling decisions, I needed to actually look at this dataset. What are the images? How are they organized? Are the annotations usable? What's going to bite me later if I don't catch it now?

The dataset is GPR B-scan radargrams from a subsurface object detection context -- three classes: intact (clean ground, no buried objects), cavities (underground voids), and utilities (buried pipes). The goal is to eventually train a model to detect anomalies in these B-scans, and this EDA is the foundation for choosing the right task framing, preprocessing, and split strategy.

---

## What We're Working With

### Dataset inventory

| Class | Original | Augmented | Total | Has Annotations |
|---|---|---|---|---|
| Intact | 75 | 900 | 975 | No |
| Cavities | 79 | 553 | 632 | Augmented only |
| Utilities | 131 | 786 | 917 | Augmented only |
| **Total** | **285** | **2,239** | **2,524** | |

Annotations exist in both YOLO format (normalized bboxes) and Pascal VOC XML format (pixel coords). I checked -- the two formats agree on object counts for every image. Good.

### Image properties

This is where it gets messy:

- **Original images** are all over the place. Intact originals are 1577x476 RGB. Cavity originals are ~588x326 RGB. Utility originals are already 224x224 RGB. No consistency at all.
- **Augmented images** are standardized: 224x224 grayscale (mode L). This is the usable set.
- The originals have NO bounding box annotations. Zero. Only the augmented set is labeled.

So in practice, the augmented set is what we train on. The originals are useful for visual reference and maybe for understanding what the augmentation pipeline did, but they can't go directly into a detection pipeline without resizing and re-annotation.

---

## What I Found

### Class balance

Not terrible, but not balanced either. Intact has ~1.5x more samples than cavities. The ratio is:
- Intact: 38.6%
- Utilities: 36.3%
- Cavities: 25.0%

This is manageable. Weighted cross-entropy or simple oversampling of cavities should handle it. Not a showstopper.

### Annotation characteristics

- Cavities: 1 object per image across the board. Single bbox per B-scan.
- Utilities: mostly 1 object, but some images have up to 3. Multi-object detection is relevant here.
- Bbox sizes vary a lot. Cavity bboxes are on the larger side (some cover 15-20% of the image area). Utility bboxes tend to be smaller and more localized.
- Spatial heatmaps show that cavities tend to appear in the center-left of the image, while utilities spread across the bottom half. There's a spatial bias here -- worth checking if the model just learns "anomaly = bottom of image" instead of actual pattern recognition.

### Intensity profiles

The classes have genuinely different pixel intensity characteristics:
- Intact images tend to be darker and more uniform. Low variance, low edge density.
- Cavities show moderate intensity with localized bright regions (the cavity reflection signature).
- Utilities have the most texture and highest edge density, consistent with the strong hyperbolic reflections from cylindrical pipes.

This is encouraging. It means even dumb features carry signal. A random forest on 9 basic image statistics (mean, std, skew, kurtosis, edge density, Laplacian variance, gradient magnitude, % zero pixels, % saturated pixels) gets decent accuracy in 5-fold CV. The exact number will vary per run, but the point is that the classes are separable even without spatial reasoning. A CNN should do considerably better.

### Depth structure

The average row profiles (mean intensity at each pixel row) show that:
- All classes have a bright band near the top -- this is the direct wave / surface reflection, standard in GPR data.
- Intact profiles decay smoothly below that.
- Cavities and utilities show secondary bright bands at mid-depth, corresponding to the buried objects.

This vertical structure is a strong prior. A model could probably benefit from knowing that anomalies live in specific depth ranges. Worth considering as a feature engineering option or as an inductive bias in the architecture.

### Frequency domain

The average 2D FFT spectra look different across classes. Utilities show more high-frequency energy (sharp edges from pipe reflections). Cavities have broader spectral content. Intact is dominated by low frequencies. This confirms that the classes have distinct spectral fingerprints, not just intensity differences.

### Augmentation quality

Each original image generates ~7-12 augmented variants. I computed SSIM (structural similarity) between augmented variants of the same original, and the values are high -- meaning the augmentations are relatively mild. The images within a group look quite similar to each other.

This has a direct consequence: **the effective diversity of this dataset is ~285 unique scenes, not 2,524.** The augmented images are variations on a theme, not independent samples. A model that memorizes 285 patterns and learns to be invariant to mild augmentations could score well on a naive random split but fail on truly new data.

---

## Red Flags

### 1. Data leakage through augmented splits

This is the biggest risk. If I split train/val/test randomly at the image level, augmented variants of the same original will land in multiple splits. The model will effectively see the test data during training (just with slightly different contrast or rotation). Validation metrics will be optimistic.

**Fix**: Split by original image ID. All augmented variants of image #42 go into the same split. This reduces effective test size but gives honest generalization estimates.

### 2. Effective sample size is small

285 unique scenes across 3 classes means ~95 unique patterns per class on average. For a detection task, that's thin. Expect high variance in results. Pre-trained backbones and careful regularization will matter.

### 3. No labels for intact = detection framing is constrained

Since intact images don't have bounding boxes (there's nothing to annotate -- they're clean), a detection model trained on this data can only learn to detect cavities and utilities. It can't learn "what absence of anomaly looks like" in a detection sense. Classification is a more natural fit for the initial experiment.

### 4. Original images are unusable without preprocessing

Different sizes, different color modes, no annotations. They need resizing, grayscale conversion, and re-annotation before they can join the training pipeline. For now, the augmented set is the path of least resistance.

### 5. JPEG compression

All images are stored as JPEG. This means lossy compression artifacts are baked into the data. For subtle GPR signal analysis this isn't ideal, but it's what the dataset provides. The compression is probably fine for classification -- the signal-to-noise ratio in these B-scans is high enough that JPEG shouldn't destroy class-discriminative features.

### 6. Spatial bias in annotations

Bounding boxes cluster in specific spatial regions per class. A lazy model might learn to predict class based on where the annotation center sits rather than what the patch actually looks like. Worth testing with spatial-ablation experiments later (e.g., random crops at different positions).

---

## Decisions and Next Steps

### Task framing: start with classification

The simplest and most data-efficient framing is 3-class image classification (intact vs cavities vs utilities) on the augmented set. This lets us:
- Use all 2,239 augmented images
- Avoid the complications of detection (anchor sizes, NMS, etc.)
- Get a quick signal on whether the model can learn these patterns at all

Once that baseline works, we move to object detection with the YOLO annotations.

### Split strategy

70/15/15 train/val/test, split by original image ID. This means:
- ~200 unique originals in train, ~43 in val, ~42 in test
- Actual image counts will be higher because of augmentation multipliers
- The split will be slightly imbalanced per class, but that's the price of honest evaluation

### Preprocessing

1. All images to grayscale (augmented already are; originals need conversion if used)
2. Normalize to [0, 1]
3. No additional augmentation in the first experiment -- the dataset is already augmented
4. If using originals later: resize to 224x224 with aspect-ratio-preserving padding

### Baseline model

Start with a simple CNN or fine-tuned ResNet18. No need for a heavy architecture on 224x224 grayscale images with 3 classes. Save the YOLO/U-Net work for the detection phase.

### Open questions

- What augmentation pipeline was used to generate the augmented set? The dataset doesn't document this. Knowing the transforms would help decide whether to add more augmentation on top.
- Are the original images from different GPR equipment or sites? If so, there might be domain shift between originals that the augmentation masks.
- How do the bounding boxes relate to the actual hyperbolic signatures in the raw GPR trace? I'd need to overlay annotations on the raw waveform data to check, but we only have images (not raw .DZT or .GPR files).
- Would a binary framing (anomaly vs. intact) perform better than 3-class? Merging cavities and utilities reduces the problem but loses information about anomaly type.

---

## Figures Generated

All saved to `results/figures/`:
- `class_distribution.png` -- bar chart and pie chart of class counts
- `image_dimension_distributions.png` -- histograms of width, height, aspect ratio
- `samples_original_*.png` -- random sample grids from each original class
- `samples_augmented_*.png` -- random sample grids from each augmented class
- `annotated_cavities.png` -- augmented cavity images with bbox overlays
- `annotated_utilities.png` -- augmented utility images with bbox overlays
- `bbox_size_distributions.png` -- width, height, area distributions for annotations
- `bbox_spatial_heatmap.png` -- where bboxes appear in image space
- `intensity_distributions.png` -- pixel intensity stats across classes
- `depth_profiles.png` -- average intensity by row (depth) per class
- `mean_images_per_class.png` -- class-average B-scans
- `fft_analysis.png` -- frequency domain comparison
- `edge_texture_analysis.png` -- Canny, Laplacian, gradient distributions
- `augmentation_diversity.png` -- how many augmentations per original
- `augmentation_ssim.png` -- structural similarity within augmented groups
- `feature_separability.png` -- 2D scatter plots of class features
