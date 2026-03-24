# Reading Notes: Accelerating Segmentation of Fossil CT Scans through Deep Learning (2024)

**Citation**: Knutsen, E.M. & Konovalov, D.A. (2024). Accelerating segmentation of fossil CT scans through Deep Learning. *Scientific Reports*, 14, 20943.
**Read**: 2026-03-23

---

## What this paper is about

Pushes the fossil CT segmentation approach further by showing you can achieve high-quality results with very little training data. Uses a UNet with EfficientNet-V2 encoder to segment Early Triassic vertebrate fossils from synchrotron micro-CT scans, training on just 1-2% of the total CT dataset. Achieves 0.96 validation Dice.

## Key technical details

**Specimen**: QMF60282, a small cemented rock fragment (~20x20x10mm) from Early Triassic Arcadia Formation, Queensland. Contains partial limb bone and cranial fragments of a procolophonid parareptile.

**CT scanning**: Synchrotron X-ray micro-CT at Australian Synchrotron, 51 keV, 2560x2560 pixels per slice, 2159 slices, 10 micron voxel size.

**Key innovation: minimal annotation.**
- Initial training: only 9 manually segmented slices (8 training, 1 validation)
- Second iteration: added 9 more manually corrected slices (18 total)
- That's it. 18 slices out of 2159 (less than 1% of data) to train a model that achieves 0.96 Dice.

**Architecture:**
- UNet segmentation model (from Yakubovskii's library on GitHub)
- EfficientNet-V2-XL as the image encoder (via timm library)
- Decoder channels: [512, 256, 128, 64, 32]
- Training crop size: 512x512 from 2560x2560 full slices
- Loss: Binary Cross Entropy
- Optimizer: AdamW with cosine annealing, weight decay 0.01
- 5000 epochs, ~10 hours on RTX 3090

**Augmentation was critical.** With only 8-17 annotated slices, augmentation was the difference between overfitting and generalization. Used the Albumentations library for random rotations (0-360 degrees) and random flips. Without augmentation, training Dice was high but validation Dice didn't improve.

**Negative sampling strategy.** For every set of 8 (or 17) fossil-containing crops, they included 2 random negative crops (no fossil). This balanced the training data between positive and negative examples.

**Test-time augmentation (TTA).** At inference, they generated 8 variations of each slice (90-degree rotations + flips based on dihedral symmetry) and averaged the predictions. This reduced noise in the predictions.

**1-channel vs 3-channel (2.5D) input.** They tested both single grayscale slices and 3-channel input including adjacent slices. Surprisingly, the 1-channel model was slightly better. This suggests the high-capacity encoder already captured enough information from a single slice.

**Iterative refinement workflow:**
1. Annotate 9 slices manually
2. Train model, predict on all slices
3. Use predictions as starting point, manually correct 9 more slices
4. Retrain on 18 slices
5. Final model achieves 0.96 Dice

This is essentially active learning / human-in-the-loop, even though they don't frame it that way.

## What matters for our project

**You don't need thousands of annotations.** 18 annotated examples with strong augmentation and a high-capacity encoder can produce 0.96 Dice. This is extremely encouraging for our GPR work where labeled data is scarce.

**The iterative refinement approach is practical.** Train on a few labels, predict, correct, retrain. We could apply this exact workflow to GPR anomaly annotations.

**EfficientNet-V2 as a backbone.** The large EfficientNet encoder brings a lot of learned visual features even though it was pretrained on natural images. The transfer from ImageNet to micro-CT fossil segmentation apparently works, which suggests transfer from ImageNet to GPR B-scans is also plausible.

**Heavy augmentation compensates for small datasets.** Random rotations through the full 360 degrees, plus flips. This is more aggressive than what's typical in GPR work.

**Negative sampling is important.** Explicitly including background/negative examples during training prevents the model from predicting fossil everywhere.

## What I want to steal

- **The minimal-annotation workflow.** We could apply the same iterative approach: label a few GPR B-scan regions, train, predict, refine.
- **Albumentations for augmentation.** We should use this library for our GPR augmentation pipeline.
- **TTA at inference.** Averaging predictions over augmented versions of the input reduces noise. Simple to implement, free performance gain.
- **Negative sampling ratio (2 negatives per 8-17 positives).** This balance worked well for them and could guide our training data preparation.
- **The UNet + EfficientNet-V2 architecture.** A strong baseline for any segmentation task with limited data.

## Limitations

- Single specimen, single locality, single preservation style. Like Yu et al. (2022), generalization to other fossils is uncertain.
- Synchrotron micro-CT is very different from field GPR data in terms of resolution, noise characteristics, and contrast. The annotation efficiency might not transfer directly.
- 5000 epochs of training is a lot. Likely significant overfitting mitigated by augmentation.
- The 0.96 Dice is a validation metric on the simplest ROI excluded from training. The average Dice across all ROIs would be lower.

## Relevance to our project: MODERATE

Not GPR, but the annotation-efficient workflow and the architectural choices are directly applicable. The key lesson is that a well-designed training pipeline can compensate for limited labels.

## 3-2-1

**3 things I learned:**
1. 18 annotated CT slices + heavy augmentation + EfficientNet-V2 encoder = 0.96 Dice segmentation. Minimal annotation can work if the architecture and augmentation are right.
2. Test-time augmentation (averaging predictions over rotated/flipped versions of each input) is a simple way to improve prediction quality.
3. The iterative annotation workflow (train on few labels -> predict -> correct -> retrain) is a practical approach to scaling up annotations.

**2 things I still don't know:**
1. Whether this minimal-annotation approach works when the target morphology is more variable (different bones, different sizes, different preservation).
2. How many annotated GPR B-scans would be needed for equivalent detection quality, given that GPR data is noisier and lower resolution than synchrotron micro-CT.

**1 thing I want to investigate:**
Test whether the iterative annotation workflow can be adapted for GPR anomaly detection: start with our existing YOLO annotations, convert to segmentation masks, train a UNet, predict on unlabeled data, and refine.
