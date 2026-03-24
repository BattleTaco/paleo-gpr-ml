# Research Log

Chronological record of research progress, decisions, and open questions for the paleo-gpr-ml project.

---

## 2026-03-23 - Initial EDA on GPR Dataset

### What I did

- Ran full exploratory analysis on the GPR B-scan dataset in notebook `01_gpr_data_exploration.ipynb`
- Cataloged all 2,524 images across 6 directories (3 original, 3 augmented)
- Parsed and validated YOLO and VOC bounding box annotations
- Computed pixel intensity distributions, depth profiles, frequency spectra
- Analyzed augmentation diversity via SSIM
- Built a simple RF classifier on image-level features as a sanity check
- Generated 16 figures saved to `results/figures/`

### What I learned

- The effective dataset is ~285 unique scenes. The 2,239 augmented images are mild transforms. This is much smaller than it looks on the surface.
- Classes are separable even with basic statistics (mean intensity, edge density, etc.). A CNN should have plenty to work with.
- Original images are a mess: variable sizes, RGB, no annotations. The augmented set is the practical training data.
- There's a real data leakage risk. Must split by original image ID, not by individual image. If augmented copies of the same scene end up in both train and val, we'll get inflated metrics that mean nothing.
- Spatial bias exists in bbox placement. The model could potentially cheat by learning location instead of pattern.
- Depth profiles are genuinely different between classes. The vertical structure of B-scans carries class-discriminative information, which makes sense physically. Cavities, utilities, and intact subsurface all interact differently with the radar pulse at different depths.

### What confused me

- Not sure what augmentation pipeline was used to create the augmented set. The dataset doesn't document this anywhere I can find. Knowing the exact transforms matters for deciding what additional augmentation to apply during training.
- The originals have no annotations at all. Did whoever created this dataset annotate from scratch on the augmented versions? Or did they annotate originals first and the annotations got lost? This is a gap in the dataset provenance that bugs me.

### Decisions made

- Start with 3-class classification on augmented images (intact vs cavities vs utilities)
- Split by original image ID (70/15/15)
- Use grayscale, normalize to [0, 1]
- Don't use original images in first experiment (too messy, no annotations)
- First model: simple CNN or fine-tuned ResNet18

### What to do next

- Build data preparation script with ID-based splits
- Write baseline experiment spec
- Start the classification baseline notebook
- Eventually: read the papers and create reading notes (Workstream A)

---

## 2026-03-23 - Baseline Classification Experiment

### What I did

- Built `src/data/build_splits.py` to create leak-proof train/val/test splits by original image ID
- Created and ran `notebooks/02_baseline_classification.ipynb` with two baselines:
  - SimpleCNN (4 conv blocks, ~500K params): 96.7% test accuracy
  - ResNet18 (pretrained, fine-tuned, ~11M params): 99.5% test accuracy
- Both trained with weighted cross-entropy, AdamW, cosine scheduling
- Generated confusion matrices, training curves, error analysis, confidence distributions

### What I learned

- Classification on this dataset is essentially solved. Both models clear the 70% bar easily, and the ResNet is near-perfect.
- The by-ID split works. No obvious leakage. Train-val gaps are tiny (SimpleCNN ~0.9%, ResNet ~0%).
- The intact class has 482 unique IDs, not 75. The augmented filenames don't directly correspond to the original directory count. More diversity than expected.
- Even the simple CNN crushes it, which means the classes are trivially separable at the image level. Not surprising given the EDA showed strong differences in intensity, texture, and depth profiles.
- The interesting challenge isn't classification on this dataset. It's detection (localizing anomalies) and generalization (working on new GPR data).

### Decisions made

- Classification baseline is done. No need to iterate further on this task.
- Next priority: object detection using YOLO annotations
- Grad-CAM analysis is worth doing to verify the model looks at the right regions
- Synthetic data planning should start soon, because the real research question is about generalization

### 3 things I learned
1. Pre-augmented datasets with distinct classes can make classification trivially easy, even with honest splits
2. Simple architectures are surprisingly strong when signal-to-noise ratio is high
3. The EDA predicted this outcome. The RF sanity check and feature separability analysis basically told us the classes are well-separated

### 2 things I still don't understand
1. Why the intact class has 482 unique IDs when there are only 75 originals. Naming convention mismatch?
2. Whether the model's near-perfect accuracy comes from learning real GPR physics or dataset-specific patterns

### 1 thing I want to investigate next
Grad-CAM on the trained ResNet18 to see what spatial regions drive predictions

---

## 2026-03-23 - Grad-CAM Analysis and Object Detection

### What I did

- Ran Grad-CAM analysis on both trained models (notebook `03_gradcam_analysis.ipynb`)
- Built detection data preparation script (`src/data/prepare_detection_data.py`)
- Trained YOLOv8n on 2-class (cavity vs utility) and binary (anomaly) detection
- Evaluated on held-out test set with by-ID splits

### What I learned

- **Cavities: the model genuinely sees the anomaly.** 51% of Grad-CAM attention falls inside the cavity bounding box, and 88% of the bbox is covered. The classification model is looking at the right region.
- **Utilities: the model uses broader context.** Only 18% of attention is inside the utility bbox. The model classifies correctly but looks at the whole image pattern, not the annotated target specifically. This makes sense because hyperbolic tails extend well beyond the bbox.
- **Detection mAP@0.5 is ~0.60.** Both 2-class (0.581) and binary (0.600) framings work. Binary is slightly better.
- **False positive rate on intact images is 0.7%.** Only 1 out of 148 intact test images triggered a false detection. The detector cleanly separates "something" from "nothing."
- **Utilities are easier to detect than cavities.** AP@0.5: utility 0.625 vs cavity 0.538. The hyperbolic signature is geometrically distinctive.

### Decisions made

- Binary anomaly detection is the better framing for fossil work (don't need to distinguish target types)
- Detection baseline is established. Ready for synthetic data experiment.
- Grad-CAM confirms the classification model learns physically meaningful features (at least for cavities)

### What to do next

- Generate synthetic fossil-like GPR data with gprMax
- Test transfer from infrastructure detection to fossil-like targets
- Write up everything so far as a coherent research narrative

---

## 2026-03-23 - Literature Review (6 Papers)

### What I did

- Read all 6 papers in `papers/references/` and wrote detailed reading notes for each in `papers/reading_notes/`
- Built out the literature matrix in `docs/literature_matrix.md` with gap analysis
- Papers covered: AI in Paleontology (review), GPR Vertebrate Skeleton Detection (Ica Desert whale), GPR Dinosaur Bones Sicily (theropod in cave), GPR AI Review (archaeological prospection), Fossil CT Segmentation (protoceratopsian skulls), Fossil CT Acceleration (minimal annotation workflow)

### What I learned

- **The intersection of ML + GPR + fossils is genuinely empty.** Archaeological GPR detection with ML exists. Fossil detection with GPR exists (physics-based). Nobody has combined ML + GPR for fossil detection. This is the gap we're filling.
- **Fossilized bone produces a positive-negative-positive polarity triplet** in GPR traces (Peredo et al.). Permittivity of mineralized bone is ~7-12 vs. ~3-5 for dry sediment. This contrast is detectable.
- **Hyperbolic signatures from bones look similar to utility hyperbolas.** The Sicily paper shows dinosaur bones producing diffraction hyperbolas that are geometrically similar to what pipes produce. This means our infrastructure GPR model might partially transfer.
- **gprMax synthetic data is standard practice** in GPR ML research. Multiple groups use it to supplement scarce real data. This validates our planned synthetic data pipeline.
- **Minimal annotations can work** if the architecture and augmentation are right. Knutsen achieved 0.96 Dice with only 18 annotated CT slices using UNet + EfficientNet-V2 + heavy augmentation.
- **Generalization across targets is hard.** Yu et al.'s CT model trained on protoceratopsian skulls completely failed on other dinosaur genera. We should expect similar brittleness and plan domain adaptation.

### Decisions made

- Our research contribution is clear: first ML-based fossil detection from GPR data, bridging infrastructure GPR training data to paleontological targets via synthetic data.
- Should compare YOLO with Faster R-CNN for detection (both used in GPR literature).
- Synthetic data pipeline with gprMax should use dielectric values from Peredo et al. and reproduce the polarity triplet signature.
- Include Hilbert transform / instantaneous amplitude as an input feature (already implemented, validated by Sicily paper).
- Use Dice and IoU for evaluation if we do segmentation, not pixel accuracy.
- Consider iterative annotation workflow from Knutsen for scaling up GPR annotations.

### What to do next

- Grad-CAM analysis on trained ResNet18 to verify it learns real GPR features
- Object detection experiment (YOLO on cavities + utilities annotations)
- Synthetic data plan with gprMax for fossil-like GPR targets
- Consider Faster R-CNN as alternative to YOLO

---

## Questions Raised by the Literature (and EDA)

- ~~What radar signatures would fossilized bone produce compared to pipes or voids?~~ **Answered**: Positive-negative-positive polarity triplet, permittivity contrast of ~7-12 vs ~3-5.
- Will synthetic fossil-like geometry actually transfer to real geology? *Still open, but synthetic data is standard in GPR ML.*
- Is object detection better than segmentation for subtle radar anomalies? *Both used in literature. Detection (R-CNN, YOLO) for B-scans, segmentation (U-Net) for C-scans.*
- ~~How much preprocessing is standard in GPR ML papers?~~ **Answered**: Migration (Kirchhoff), Hilbert transform, gain correction. Varies by study.
- ~~Do most papers use raw radargrams or processed radargrams?~~ **Answered**: Both. Some detect on raw B-scans, others on migrated. No consensus.
- Would a binary anomaly framing work better than 3-class for the initial task? *Still open.*
- How should we handle the domain gap between synthetic and real data when we get there? *Still open, but fine-tuning on real data after synthetic pretraining is the common approach.*

---

## Methods to Borrow from the Literature

- **Preprocessing techniques**: Kirchhoff migration (collapses hyperbolas to points), Hilbert transform / instantaneous amplitude (highlights reflectors regardless of polarity), AGC gain correction
- **Augmentation ideas**: Full 360-degree rotation + flips (Knutsen), Albumentations library, negative sampling (2 negatives per 8-17 positives)
- **Detection framing approaches**: Faster R-CNN for B-scan detection (Verdonck, Pham), YOLO for object detection, U-Net for time-slice segmentation (Kücükdemirci)
- **Evaluation metrics beyond accuracy**: Dice coefficient, IoU, per-class recall, confusion matrices. Accuracy is misleading when background dominates.
- **Synthetic data strategies (gprMax)**: Generate synthetic B-scans with known target geometry and dielectric properties. Use for pretraining/hyperparameter tuning, fine-tune on real data. Include polarity triplet signature as physics constraint.
- **Annotation efficiency**: Iterative human-in-the-loop (Knutsen): label few -> train -> predict -> correct -> retrain. Test-time augmentation for free performance.
