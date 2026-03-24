# Literature Matrix

Structured comparison of papers relevant to GPR-based fossil prospecting with ML.

## Core Papers

| Paper | Year | Domain | Data Type | Task | Best Model | Key Metric | Training Data Size | Synthetic Data? | Relevance |
|---|---|---|---|---|---|---|---|---|---|
| Yu et al. - AI in Paleontology | 2024 | Paleo+AI | Mixed | Review | N/A | N/A | N/A | N/A | HIGH |
| Peredo et al. - Vertebrate Skeleton GPR | 2024 | GPR+Paleo | 400 MHz B-scans | Detection | Physics (trace analysis) | Qualitative | N/A | Forward modeled | VERY HIGH |
| Catanzariti et al. - Dinosaur GPR Sicily | 2023 | GPR+Paleo | 2 GHz B-scans | Detection | Signal processing (Kirchhoff migration) | Qualitative | N/A | No | VERY HIGH |
| Kücükdemirci & Sarris - GPR AI Review | 2022 | GPR+Archaeology | B-scans + C-scans | Review | Various (R-CNN, U-Net, ResNet) | Up to 94% acc | ~1000 real + synthetic | Yes (gprMax) | HIGH |
| Yu et al. - Fossil CT Segmentation | 2022 | Paleo+CV | Micro-CT slices | Segmentation | DeepLab v3+ (MobileNet v1) | 0.894 Dice | 7986 slices | No | MODERATE-HIGH |
| Knutsen & Konovalov - Fossil CT Acceleration | 2024 | Paleo+CV | Synchrotron micro-CT | Segmentation | UNet (EfficientNet-V2-XL) | 0.96 Dice | 18 slices (!) | No | MODERATE |

## Key Methods Across Papers

| Method | Used By | Our Applicability |
|---|---|---|
| gprMax synthetic data | Green et al. (via GPR AI Review), multiple B-scan studies | Core to our synthetic data pipeline |
| Forward modeling (Ricker wavelet) | Peredo et al. (Vertebrate Skeleton GPR) | Validate synthetic signatures |
| Kirchhoff migration | Catanzariti et al. (Dinosaur Sicily) | Preprocessing option for detection |
| Hilbert transform / instantaneous amplitude | Catanzariti et al. (Dinosaur Sicily) | Already implemented in our features |
| U-Net segmentation | Yu et al. (CT), Knutsen (CT), Kücükdemirci (GPR C-scans) | Candidate for GPR anomaly segmentation |
| DeepLab v3+ | Yu et al. (CT) | Candidate architecture |
| Faster R-CNN | Verdonck, Pham et al. (via GPR AI Review) | Compare with YOLO for detection |
| ResNet152 transfer learning | Green et al. (via GPR AI Review) | Our ResNet18 already works; deeper may help for harder tasks |
| Iterative annotation (human-in-the-loop) | Knutsen & Konovalov (CT) | Practical for scaling GPR annotations |
| Test-time augmentation | Knutsen & Konovalov (CT) | Free performance boost at inference |
| Polarity triplet analysis | Peredo et al. (Vertebrate Skeleton GPR) | Physics constraint for synthetic data |

## Gap Analysis: What Nobody Has Done

1. **ML-based fossil detection from GPR data.** Archaeological GPR detection exists. Fossil GPR detection exists but only via manual/physics-based interpretation. ML + GPR + fossils = unstudied.

2. **Synthetic-to-real transfer for GPR fossil targets.** Synthetic GPR data has been used for infrastructure targets. Synthetic fossil-like targets in GPR have not been attempted.

3. **Domain adaptation from infrastructure GPR to paleontological GPR.** Infrastructure GPR datasets are plentiful. Paleontological GPR data is rare. Bridging this gap with transfer learning or domain adaptation is an open problem.

4. **Cross-frequency GPR detection.** Models trained on 400 MHz data being tested on 2 GHz data (or vice versa) has not been systematically studied.

## What This Means for Our Project

The literature says:
- GPR can physically detect buried fossils (Peredo, Catanzariti). The physics works.
- ML can detect objects in GPR B-scans (Green, Verdonck, Pham). The method works.
- Nobody has combined these two things. The intersection is empty.

Our contribution:
- Train ML model on infrastructure GPR data (where we have annotations)
- Generate synthetic fossil-like GPR data with gprMax using dielectric properties from the paleontological GPR papers
- Test whether the model transfers to detecting fossil-like anomalies
- Use domain adaptation if direct transfer fails

This is a genuine gap. Not "incremental improvement on existing method" but "first application of existing method to new domain."
