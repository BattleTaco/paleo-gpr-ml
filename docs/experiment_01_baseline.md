# Experiment 01: GPR B-scan Classification Baseline

## Objective

Determine whether a CNN can reliably distinguish between intact ground, cavities, and utilities in GPR B-scan radargrams. This is the simplest possible task framing for the dataset and serves as a proof-of-concept before moving to detection.

## Dataset

- Source: Augmented GPR B-scans only (2,239 images total)
  - `augmented_intact/`: 900 images
  - `augmented_cavities/`: 553 images
  - `augmented_utilities/`: 786 images
- Format: 224x224 grayscale JPEG
- Original images excluded (inconsistent sizes, no annotations, RGB)

## Task Type

3-class image classification: intact vs. cavities vs. utilities

## Preprocessing

- Load as grayscale (already grayscale for augmented set)
- Normalize pixel values to [0, 1]
- No additional augmentation in v1 (dataset is already augmented)
- Consider adding in v2: random horizontal flip, slight rotation (+-5 deg), brightness jitter

## Train/Val/Test Split

- 70/15/15 split BY ORIGINAL IMAGE ID
- Critical: all augmented variants of the same original go in the same split. This prevents data leakage from augmented near-duplicates.
- Effective unique scenes: ~200 train, ~43 val, ~42 test (approximate)
- Actual image counts will be higher due to augmentation multiplier
- Split should be stratified by class to maintain proportions

## Baseline Models

Two baselines to compare:

### Baseline A: Simple CNN

- Architecture: 3-4 conv blocks (conv -> batch norm -> relu -> max pool), then FC layers
- Input: 1x224x224
- Output: 3 classes
- Dropout: 0.3-0.5
- Why: establishes a lower bound. If this works well, the task might be too easy for this dataset.

### Baseline B: Fine-tuned ResNet18

- Architecture: torchvision ResNet18, pretrained on ImageNet
- Modify first conv layer to accept 1 channel (average the pretrained weights or use single channel)
- Replace final FC layer for 3 classes
- Freeze backbone for first 5 epochs, then unfreeze
- Why: transfer learning is standard practice and should give a reasonable upper bound for this dataset size

## Training

- Optimizer: AdamW
- Learning rate: 1e-3 (simple CNN), 1e-4 (ResNet18 fine-tuning)
- Scheduler: cosine annealing
- Epochs: 30-50 (with early stopping, patience 7)
- Batch size: 32
- Loss: cross-entropy with class weights (inverse frequency)
- Seed: 42

## Evaluation Metrics

- Accuracy (overall and per-class)
- Precision, recall, F1 (macro and per-class)
- Confusion matrix
- ROC-AUC (one-vs-rest)
- Training/validation loss curves

## Success Criteria

- **Minimum**: >70% accuracy on held-out test set (by-ID split). This would confirm that the classes are learnable.
- **Good**: >85% accuracy with reasonable per-class recall (no class completely ignored).
- **Concerning**: >95% accuracy would suggest possible leakage or the task being trivially easy with this data.

## Known Risks

1. **Data leakage**: If the split isn't done by original ID, results will be inflated. This is the single most important thing to get right.
2. **Small effective dataset**: ~285 unique scenes. High variance in results is expected. Should report mean +/- std over multiple seeds or folds.
3. **Overfitting**: With limited diversity, the model may memorize training patterns. Monitor train-val gap closely.
4. **Augmentation artifacts as shortcuts**: If augmentation introduced class-correlated artifacts (e.g., specific contrast levels), the model might learn artifacts instead of GPR patterns. Check with Grad-CAM or similar.
5. **Class imbalance**: Cavities are underrepresented (~25%). Weighted loss should help but monitor per-class recall.

## What This Experiment Does NOT Answer

- Whether the model can detect anomalies in new, unseen GPR data from different equipment or sites
- Whether bounding box detection works (that's experiment 02)
- Whether synthetic data can supplement real data
- Whether the model learns physically meaningful GPR features vs. dataset-specific shortcuts

## Deliverables

- Trained model checkpoints in `results/models/`
- Training logs in `results/logs/`
- Confusion matrix and metric plots in `results/figures/`
- Summary metrics table in `results/tables/`
- Experiment notes added to `docs/research_log.md`
