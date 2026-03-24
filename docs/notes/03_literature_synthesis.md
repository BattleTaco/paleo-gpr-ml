# Literature Synthesis: What the Papers Tell Us

**Date**: 2026-03-23
**Papers reviewed**: 6 (see `papers/reading_notes/` for individual notes)

---

## The state of the field

There are three active research threads that our project sits at the intersection of:

**Thread 1: ML for GPR interpretation.** This is mostly in civil engineering and archaeology. People use CNNs (R-CNN, YOLO, ResNet) and segmentation models (U-Net) to detect buried infrastructure, graves, and archaeological features in B-scans and C-scans. The methods work well when you have enough labeled data. Synthetic data from gprMax is commonly used to fill the gap. Best results: ~94% accuracy for grave detection (Green), 92% Dice for archaeological segmentation (Kücükdemirci).

**Thread 2: GPR for fossil detection.** This is much thinner. Two papers actually detect fossils with GPR: a whale skeleton in Peru (Peredo, 2024) and dinosaur bones in Sicily (Catanzariti, 2023). Both use physics-based interpretation, not ML. They confirm that GPR can physically detect buried fossils through dielectric contrast, but the interpretation requires expert human judgment.

**Thread 3: DL for paleontological image analysis.** Mostly CT segmentation. Yu et al. (2022) and Knutsen (2024) both use U-Net variants to segment fossils from CT scans. The methods work well within-domain but fail to generalize across taxa. Data scarcity and annotation cost are the universal bottlenecks.

**The gap**: Nobody has applied ML to GPR data for fossil detection. Threads 1 and 2 are parallel lines that don't intersect. Our project connects them.

## What we know works

1. **CNNs can detect objects in GPR B-scans.** Multiple groups have demonstrated this for infrastructure and archaeological targets. The visual patterns (hyperbolas, bright patches, linear features) are learnable.

2. **Fossils produce detectable GPR signatures.** Bone permittivity (~7-12) contrasts with sediment (~3-5). The polarity triplet (positive-negative-positive) is a diagnostic feature. Hyperbolic diffractions occur from bone surfaces.

3. **Synthetic GPR data supplements real data.** gprMax is the standard tool. Synthetic data works for pretraining and hyperparameter tuning. The gap between synthetic and real remains a challenge.

4. **Transfer learning works for GPR.** ImageNet-pretrained models fine-tuned on GPR data outperform models trained from scratch, at least when real data is limited (Green: ResNet152 on ~1000 images).

5. **Minimal annotations can go far.** Knutsen showed 0.96 Dice from 18 annotated slices. The key ingredients: high-capacity encoder (EfficientNet), heavy augmentation, iterative refinement.

## What we don't know

1. **Whether infrastructure GPR patterns transfer to fossil detection.** Utility hyperbolas and bone hyperbolas look geometrically similar, but the scale, frequency, noise, and surrounding medium are different. Nobody has tested this transfer.

2. **How realistic gprMax fossil simulations are.** We can simulate fossil-like targets with the right dielectric values, but real geology is messy: heterogeneous soil, surface roughness, clay layers, water table effects. The simulation-to-reality gap is uncertain.

3. **Whether detection or segmentation is better for GPR anomalies.** Detection (bounding boxes) is simpler and matches our existing annotations. Segmentation could be more precise but requires pixel-level labels we don't have.

4. **The effect of GPR frequency on ML detection.** Our dataset is from some unknown frequency. The paleontological papers use 400 MHz and 2 GHz. A model trained on one frequency may not work on another.

## The novel contribution

Our project's novelty is clear and defensible:

**First ML-based approach to fossil detection in GPR data.** We're not claiming a new architecture or a new training technique. We're applying existing methods (YOLO/R-CNN, synthetic data, transfer learning) to a new problem (fossil prospecting from GPR) where no prior ML work exists.

The research questions:
1. Can a model trained on infrastructure GPR data detect fossil-like anomalies?
2. Can synthetic GPR data with fossil-like targets improve detection?
3. What is the domain gap between infrastructure GPR and paleontological GPR, and can it be bridged?

These are answerable with the tools we have. The answers have practical value for paleontological fieldwork.

## Methodological lessons from the literature

**For our detection experiment:**
- Use both YOLO and Faster R-CNN, compare (GPR AI Review)
- Report Dice/IoU, not just accuracy (CT papers)
- Include Hilbert transform as an input feature (Sicily paper)
- Use weighted loss for class imbalance (already doing this)

**For our synthetic data pipeline:**
- Use gprMax with dielectric values from Peredo: bone ~7-12, sediment ~3-5 (Vertebrate Skeleton paper)
- Reproduce the polarity triplet signature as a validation check (Vertebrate Skeleton paper)
- Model targets at multiple depths and in different host materials (general best practice)
- Use synthetic data for pretraining, fine-tune on real data (GPR AI Review)

**For evaluation and generalization:**
- Expect domain shift to be a problem (CT segmentation papers both show poor generalization)
- Plan for iterative refinement when moving to new data (Knutsen workflow)
- Test on completely different GPR data if possible

## Open research directions beyond V1

- 3D GPR volumes with 3D CNNs (suggested by GPR AI Review but unexplored)
- Cross-frequency transfer learning (400 MHz <-> 2 GHz)
- Active learning for GPR annotation (adapt Knutsen's iterative approach)
- Multi-modal fusion: GPR + magnetometry or GPR + resistivity
- Comparison with traditional geophysical interpretation by domain experts
