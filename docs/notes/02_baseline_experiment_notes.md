# Baseline Classification Experiment Notes

**Date**: 2026-03-23
**Notebook**: `notebooks/02_baseline_classification.ipynb`
**Experiment spec**: `docs/experiment_01_baseline.md`

---

## Setup

- Task: 3-class classification (intact vs. cavities vs. utilities)
- Data: 2,239 augmented GPR B-scans, 224x224 grayscale
- Split: 70/15/15 by original image ID (1521 train / 329 val / 389 test)
- No augmentation during training (data is pre-augmented)
- Weighted cross-entropy loss (inverse frequency)
- Two baselines: SimpleCNN (4 conv blocks) and ResNet18 (pretrained, fine-tuned)

## Results

| Model | Test Accuracy | Test Loss | Epochs | Train-Val Gap |
|---|---|---|---|---|
| SimpleCNN | 96.7% | 0.122 | 19 | 0.009 |
| ResNet18 | 99.5% | 0.014 | 15 | 0.000 |

Both models cleared the 70% minimum bar by a wide margin. The ResNet18 is nearly perfect.

## What happened

The task turned out to be significantly easier than expected, even with the by-ID split. A few things likely contribute:

1. **The classes are visually very distinct.** The EDA already showed this. Intact images have a smooth, uniform appearance. Cavities have localized bright patches. Utilities have strong hyperbolic features with high edge density. Even basic image statistics (mean, std, edge density) gave good RF accuracy. A CNN should crush this.

2. **The by-ID split didn't hurt as much as feared.** I expected a big drop from the augmented variant similarity, but the 482 unique IDs for intact (not 75 as originally assumed) means the effective training set is larger than I initially thought. The intact class alone has more unique scenes than I estimated.

3. **224x224 is plenty of resolution.** The discriminative features are macroscopic (bright bands, hyperbolas, texture changes), not pixel-level details. The model doesn't need to look at fine structure to distinguish these classes.

## Is this too easy?

Honestly, probably yes for the classification framing. A few things to consider:

**Arguments that this is legitimately good:**
- The split is by original ID, so there's no obvious leakage path
- The train-val gap is tiny, suggesting the model is generalizing, not memorizing
- The 96.7% accuracy of the simple CNN means even a shallow architecture picks this up

**Arguments that we should be cautious:**
- 99.5% on a research dataset with pre-applied augmentation is suspiciously high
- The augmented variants of different originals within the same class may still be quite similar (e.g., all intact images look roughly the same regardless of original)
- We haven't checked whether the model is learning GPR features or dataset-specific shortcuts (color distribution, border artifacts, etc.)
- This dataset has 3 very different classes. Real-world GPR anomaly detection would need to distinguish much subtler targets (fossil-like objects from rocks, roots, and geological features)

**What would make me more confident:**
- Grad-CAM showing the model focuses on the actual anomaly regions, not image borders
- Testing on a completely different GPR dataset (different equipment, different site)
- A harder task framing (detection instead of classification, or new classes)

## What the model gets wrong

With only ~2 misclassified images for ResNet18, there's not much to analyze. Looking at the misclassified samples, the errors seem to be on genuinely ambiguous images where the anomaly signature is weak or unusual. The confidence distribution shows the model is generally confident when right and less confident when wrong, which is healthy behavior.

The SimpleCNN errors are slightly more interesting. It has about 13 misclassifications, and the pattern seems to be confusion between cavities and intact when the cavity signature is faint.

## Implications for the project

1. **Classification on this dataset is basically solved.** Not much point in iterating on classification architectures or hyperparameters. The signal-to-noise ratio is too high for this to be interesting.

2. **Move to detection.** The YOLO annotations exist for cavities and utilities. Object detection is harder because the model needs to localize, not just classify. This is where the real challenge lives.

3. **The bigger question is generalization.** Can a model trained on this dataset detect anomalies in GPR data from a different source? That's the transfer learning and synthetic data question, and it's the core of this research project.

4. **Grad-CAM is still worth doing.** Even though accuracy is high, verifying that the model attends to the right spatial regions builds confidence that we're learning physics, not artifacts.

## Next steps

- Run Grad-CAM on the ResNet18 to visualize attention
- Start experiment 02: YOLO-based object detection on annotated cavities + utilities
- Begin the synthetic data planning (gprMax) to address the generalization question
- Read the papers to understand what detection architectures work well for GPR

## Lessons

- High accuracy on a pre-augmented dataset with distinct classes doesn't tell us much about real-world performance. The EDA was more informative than the model training for understanding this data.
- The by-ID split worked as intended. No sign of leakage.
- The intact class has way more unique IDs (482) than originals (75). The augmented filenames aren't a 1:1 map to the original directory. This is good for diversity but confusing for provenance tracking.
- Simple CNNs are surprisingly strong when the classes are visually distinct. The ResNet's marginal gain (96.7% to 99.5%) suggests that the extra capacity mostly helps on edge cases.
