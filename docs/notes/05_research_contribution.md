# Research Contribution: What Makes This Novel

**Date**: 2026-03-23

---

## The gap

After reading the literature, the picture is clear. Three things exist independently:

1. **GPR can detect buried fossils.** Demonstrated by Peredo et al. (whale skeleton, Peru) and Catanzariti et al. (theropod bones, Sicily). Both used physics-based interpretation, not ML.

2. **ML can detect objects in GPR B-scans.** Demonstrated by multiple groups for infrastructure (pipes, cavities, graves). Uses CNN-based detection (YOLO, R-CNN) with synthetic and real training data.

3. **DL is useful in paleontology.** Demonstrated for CT segmentation, fossil classification, morphometric analysis. Data scarcity is the bottleneck.

**What doesn't exist**: ML-based fossil detection from GPR data. Nobody has connected (1) and (2) for the paleontological use case.

## Our contribution

**First ML approach to fossil prospecting from GPR radargrams.**

The research pipeline:
1. Train anomaly detection on infrastructure GPR data (where labels exist)
2. Generate synthetic GPR data with fossil-like targets using gprMax (with dielectric parameters from the paleontological GPR papers)
3. Test transfer from infrastructure anomalies to fossil-like anomalies
4. Characterize the domain gap and explore adaptation strategies

This isn't a new architecture or a new training technique. It's a new *application* of existing methods to an unstudied problem, informed by the physics of GPR fossil signatures.

## Why this matters

**Paleontological fieldwork is slow and expensive.** Finding fossils currently relies on surface exposure (erosion, weathering) or luck. Subsurface prospecting with GPR is physically possible (the papers prove it) but requires expert interpretation of every B-scan. An ML system that pre-screens GPR data for fossil-like anomalies would:

- Reduce the volume of data that needs expert review
- Flag high-priority areas for targeted excavation
- Enable systematic subsurface surveys of large areas

**Nobody is working on this.** The paleontology community doesn't have ML expertise in GPR. The GPR ML community doesn't target paleontological applications. We sit at the intersection.

## What we need to show

To make the contribution credible, we need:

1. **Baseline detection works on infrastructure data.** (Experiment 02) Shows the method is sound.
2. **Synthetic fossil data is physically realistic.** (gprMax validation) Shows the synthetic data captures the right physics.
3. **Some degree of transfer from infrastructure to fossil-like targets.** Even partial transfer is interesting. If it fails completely, characterizing *why* is also valuable.
4. **Analysis of what the model learns.** (Grad-CAM, feature analysis) Shows the model uses physically meaningful features, not dataset artifacts.

## What we don't need to show

- State-of-the-art detection on infrastructure GPR (others have done this, it's not our contribution)
- Perfect fossil detection (this is a first attempt, baseline results are fine)
- Field validation on real fossils (aspirational but not required for the first paper)

## Framing for a paper

**Title concept**: "Toward ML-Based Fossil Prospecting: Detecting Subsurface Anomalies in GPR Radargrams with Synthetic Training Data"

**Story**:
- GPR is physically capable of detecting buried fossils (cite Peredo, Catanzariti)
- ML detection of GPR anomalies is established for infrastructure (cite Green, Kücükdemirci, others)
- We bridge these by training on infrastructure data, generating synthetic fossil-like targets, and testing transfer
- Results: [baseline detection performance, Grad-CAM analysis, synthetic data quality, transfer results]
- This is the first study to apply ML to GPR data for paleontological prospecting

**Venue options**:
- Geophysics journals: *Geophysics*, *Near Surface Geophysics*
- Remote sensing: *Remote Sensing*, *IEEE TGRS*
- Paleontology + tech: *Frontiers in Earth Science*, *Computers & Geosciences*
- Interdisciplinary: *Scientific Reports*

## Risks to novelty

- Someone else publishes this exact idea first. Risk is low given how niche the intersection is, but the GPR community is active.
- The transfer doesn't work at all, and we can't show any positive result on fossil-like targets. Mitigation: characterizing the failure mode is itself publishable if done well.
- Reviewers don't see the practical value. Mitigation: frame around the fieldwork impact and the broader trend of AI in earth sciences.
