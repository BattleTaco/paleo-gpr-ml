# Data Understanding: GPR B-scan Dataset

## Dataset source

GPR (ground-penetrating radar) B-scan radargrams from a subsurface object detection context. Three classes representing different subsurface conditions: intact ground (no buried objects), cavities (underground voids), and utilities (buried pipes/conduits).

The data appears to come from a Mendeley-hosted dataset. No README or provenance file was included with the download.

## Dataset path in repo

```
data/raw/GPR_data/
├── intact/                    # 75 original intact B-scans
├── augmented_intact/          # 900 augmented intact B-scans
├── cavities/                  # 79 original cavity B-scans
├── augmented_cavities/        # 553 augmented cavity B-scans
│   └── annotations/
│       ├── Yolo_format/       # 553 .txt files
│       └── VOC_XML_format/    # 553 .xml files
├── Utilities/                 # 131 original utility B-scans
└── augmented_utilities/       # 786 augmented utility B-scans
    └── annotations/
        ├── YOLO_format/       # 786 .txt files
        └── VOC_XML_format/    # 786 .xml files
```

Note the inconsistency: `Yolo_format` vs `YOLO_format` between cavity and utility annotation dirs.

## File formats

**Images**: JPEG format throughout. Originals use `.JPG` extension (uppercase), augmented use `.jpg` (lowercase). This doesn't matter functionally but it's a sign the original and augmented sets were prepared differently.

**YOLO annotations**: One `.txt` per image. Each line is `class_id center_x center_y width height` with all coordinates normalized to [0, 1]. Class ID is always 0 (single class per annotation set).

**VOC XML annotations**: Standard Pascal VOC format. Each `.xml` contains image dimensions (always 224x224x3, though the augmented images are actually grayscale), class name (`cavities` or `utilities`), and bounding box in pixel coordinates (`xmin`, `ymin`, `xmax`, `ymax`).

Both annotation formats agree on object counts for every image. I verified this in the EDA.

## Number of samples

| Class | Original | Augmented | Total | % of Dataset |
|---|---|---|---|---|
| Intact | 75 | 900 | 975 | 38.6% |
| Cavities | 79 | 553 | 632 | 25.0% |
| Utilities | 131 | 786 | 917 | 36.3% |
| **Total** | **285** | **2,239** | **2,524** | |

Augmentation ratios: intact 12x, cavities 7x, utilities 6x. The augmentation was applied more aggressively to intact, probably to balance the dataset.

**Effective unique scenes: ~285.** The augmented images are mild transforms of the originals (high intra-group SSIM). The true diversity of this dataset is much lower than the 2,524 count suggests.

## Label structure and implications for modeling

**What has labels:**
- `augmented_cavities`: YOLO + VOC bounding boxes. 1 object per image.
- `augmented_utilities`: YOLO + VOC bounding boxes. 1-3 objects per image.

**What does NOT have labels:**
- `intact` (original): no annotations. Nothing to annotate.
- `augmented_intact`: no annotations. Same reason.
- `cavities` (original): no bounding boxes.
- `utilities` (original): no bounding boxes.

This means:
- **Classification** is possible for all augmented images (3-class, using directory as label).
- **Detection** is possible only for augmented cavities + utilities (need to treat intact as background/negative).
- **Segmentation** would require converting bboxes to masks, which is doable but loses some information.

The natural first task is classification. Detection comes second.

## Visual patterns and likely anomaly signatures

**Intact**: Relatively uniform, smooth vertical intensity gradient. Bright band at top (surface reflection / direct wave), then gradual darkening with depth. Low texture, low edge density. These are the "boring" scans where nothing is buried.

**Cavities**: Show localized bright regions at mid-depth. The cavity produces a contrast change because the void has different dielectric properties than the surrounding soil. Some cavity signatures look like diffuse bright patches rather than clean hyperbolas. Moderate texture.

**Utilities**: Show the strongest and most distinctive patterns. Buried pipes produce classic hyperbolic reflections. The hyperbola apex marks the pipe center, and the shape encodes depth and wave velocity. These have the highest edge density and gradient magnitudes of any class.

All classes share a bright horizontal band in the top few rows (the direct/surface wave). This is standard in GPR data and not class-discriminative. A model that only looks at the top rows will fail.

**Spatial bias**: Cavity bboxes tend to cluster center-left. Utility bboxes spread across the bottom half. A model could potentially learn spatial shortcuts instead of actual GPR features.

## Data quality issues and risks

1. **Original image inconsistency**: Intact originals are 1577x476 RGB. Cavity originals are ~588x326 RGB. Utility originals are 224x224 RGB. Three completely different formats. The augmented set normalizes this to 224x224 grayscale, which is what we'll actually use.

2. **JPEG compression**: All images are lossy JPEG. Fine-grained signal information is degraded. Probably not a problem for classification, but worth noting for any attempt at signal-level analysis.

3. **Augmentation provenance unknown**: No documentation of what transforms were applied. From SSIM analysis, the augmentations are relatively mild (variants of the same original look very similar). Likely simple geometric + contrast transforms.

4. **Data leakage risk**: Augmented variants of the same original image are near-duplicates. If they end up in different train/test splits, the model effectively sees its test data during training. Split by original image ID is mandatory.

5. **Small effective sample size**: ~95 unique originals per class on average. Results will have high variance. Expect overfitting unless regularized properly.

6. **Potential spatial shortcut**: Bounding boxes cluster in class-specific regions. A position-aware model might learn "object at y=150 means utility" instead of learning actual GPR signatures.

## Implications for preprocessing

- Convert everything to grayscale (augmented already is; originals would need it if used)
- Normalize to [0, 1] or [-1, 1]
- No additional augmentation needed for first experiment (dataset is pre-augmented)
- If using originals: resize to 224x224 with care (aspect ratios differ wildly)
- Consider per-image standardization (zero mean, unit variance) as an alternative to simple [0, 1] scaling

## Implications for modeling

- **Start with classification**: 3-class on augmented set. Simplest, uses all data.
- **Split by original ID**: This is non-negotiable. Random image-level splits will leak.
- **Expect ~285 effective training examples**: Plan for high regularization, dropout, weight decay. Pre-trained models will help.
- **Class imbalance is moderate**: Cavities at 25% is a bit low. Use weighted loss (inverse frequency) or oversample.
- **Watch for overfitting**: Track train-val gap carefully. If train accuracy hits 99% while val is at 75%, we're memorizing.
- **Grad-CAM or similar**: After training, check what the model actually looks at. If it's learning border artifacts or spatial position rather than GPR features, the results are unreliable.

## Open questions

- What augmentation pipeline produced the augmented set? Knowing this matters for deciding whether to add more augmentation.
- Are the original images from the same GPR equipment and survey site? Or is there domain shift between originals?
- How well do the bounding boxes align with the actual physical anomaly signatures? Are they tight or loose?
- Would a binary framing (anomaly vs. intact) give better results than 3-class?
- What happens when we test on completely new GPR data from a different source?
