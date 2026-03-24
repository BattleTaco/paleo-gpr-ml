# Synthetic Data Plan: gprMax Fossil-like GPR Targets

**Date**: 2026-03-23
**Status**: Planning
**Depends on**: Literature review (done), baseline experiment (done)

---

## Motivation

Real GPR data with labeled fossil targets essentially doesn't exist. Our infrastructure dataset has cavities and utilities, not fossils. To test whether ML can detect fossil-like anomalies in GPR data, we need synthetic data that simulates what a buried fossil would look like to a GPR antenna.

The literature validates this approach. Multiple GPR ML papers use gprMax synthetic data to supplement real training data (Green et al. for graves, Ali et al. for geometric shapes, others cited in Kücükdemirci & Sarris 2022).

## What gprMax does

gprMax is an open-source FDTD (finite-difference time-domain) electromagnetic wave simulation tool. You define:
- A 2D or 3D model space with materials (dielectric properties)
- A source waveform (typically Ricker wavelet)
- Transmitter and receiver positions
- The simulation computes the full electromagnetic wave propagation

Output: synthetic A-scans (traces) that can be assembled into B-scans.

## Physical parameters from the literature

From Peredo et al. (2024) - Vertebrate Skeleton GPR:
- Mineralized bone permittivity: ~7-12 (depends on mineralization degree)
- Dry sand/sediment permittivity: ~3-5
- Wet sediment permittivity: ~10-25
- The contrast ratio matters more than absolute values

From Catanzariti et al. (2023) - Dinosaur Bones Sicily:
- Limestone matrix permittivity: ~4-8
- The 2 GHz frequency resolves features at centimeter scale
- Bone in limestone produces clear diffraction hyperbolas

General GPR material properties:
- Air: permittivity = 1
- Water: permittivity = 80
- Clay (dry): permittivity = 3-6
- Clay (wet): permittivity = 10-40
- Rock (dry): permittivity = 4-6

## Synthetic target design

### Target types to simulate

1. **Isolated bone fragment** (simplest case)
   - Elliptical cross-section, 5-20 cm major axis
   - Permittivity 7-12
   - Depth: 0.3-1.5 m
   - Host: dry sand (3-5) or clay (5-10)

2. **Multiple bone fragments** (scatter pattern)
   - 3-8 elliptical targets in a cluster
   - Variable sizes (2-15 cm)
   - Variable depths within a layer
   - Simulates a disarticulated skeleton

3. **Articulated skeleton approximation**
   - Connected rectangular/elliptical segments
   - Variable cross-section along length
   - Depth: 0.5-2.0 m
   - Simulates a partial skeleton in situ

4. **Fossil in sedimentary layer**
   - Bone target within a horizontal layer of different permittivity
   - The layer interface creates additional reflections
   - More realistic than a target in homogeneous background

5. **Control targets (for comparison)**
   - Cavity (air-filled void): permittivity = 1
   - Pipe (metal): high conductivity
   - Rock (non-fossil): permittivity similar to surrounding sediment

### Source parameters

- **Frequency**: 400 MHz (matches Peredo) and 800 MHz (common survey frequency)
- **Waveform**: Ricker wavelet (standard in gprMax)
- **Antenna spacing**: 0.025-0.05 m trace interval
- **Time window**: adjusted per depth of interest

### Host medium variations

To avoid the model learning a single background pattern:
- Dry sand (permittivity 3-5, conductivity 0.001 S/m)
- Wet sand (permittivity 15-25, conductivity 0.01 S/m)
- Dry clay (permittivity 5-8, conductivity 0.01 S/m)
- Loam/mixed soil (permittivity 8-15, conductivity 0.02 S/m)
- Add random small-scale heterogeneity (fractal noise on permittivity)

### Planned dataset size

- 50-100 gprMax models per target type
- Each model produces one synthetic B-scan
- 5 target types x ~75 models = ~375 synthetic B-scans
- Plus ~100 "background only" (no target) B-scans
- Total: ~475 synthetic B-scans

This is modest but matches what others have done. Green et al. used ~1000 images total (real + synthetic). We'll have ~475 synthetic + 2239 real infrastructure images.

## Validation checks

Before using synthetic data for training:

1. **Visual plausibility**: Do the synthetic B-scans look like real GPR data? Compare side by side.
2. **Polarity triplet**: Verify that bone targets produce the positive-negative-positive pattern described by Peredo.
3. **Hyperbola shape**: Compare synthetic hyperbolas from bone with real hyperbolas from utilities. They should be geometrically similar (scaled by velocity).
4. **Noise level**: Add realistic noise to synthetic data. Real GPR has system noise, ground clutter, and interference.
5. **Amplitude consistency**: Normalize synthetic and real data to the same scale.

## Training strategy

**Phase 1: Infrastructure-only baseline**
- Train YOLO on real cavity/utility data
- Evaluate on held-out test set
- This is our detection baseline (Experiment 02)

**Phase 2: Synthetic pretraining**
- Pretrain YOLO on synthetic data (all target types including fossil-like)
- Fine-tune on real infrastructure data
- Compare with Phase 1 baseline
- Does synthetic pretraining help, hurt, or make no difference?

**Phase 3: Transfer to fossil targets**
- Take the Phase 2 model
- Test on synthetic fossil-only B-scans (held out from training)
- Can it detect fossil-like targets it was pretrained on but never saw during fine-tuning?

**Phase 4 (aspirational): Real paleontological GPR data**
- If we can get any real GPR data from a paleontological site, test the model
- This is the ultimate validation but requires external data we don't have yet

## Dependencies and timeline

- **gprMax installation**: Need to install and test locally. FDTD simulations can be compute-heavy.
- **Model design**: Script to generate gprMax input files programmatically (parameterized by target type, depth, host medium)
- **Post-processing**: Convert gprMax output to B-scan images in the same format as our real data
- **Integration**: Add synthetic data to our data pipeline alongside real data

This should start after the Grad-CAM analysis and detection baseline are done, since those inform what patterns the model actually learns and how hard detection is on real data.

## Risks

1. **Simulation fidelity**: gprMax simulates idealized conditions. Real soil is messy. The synthetic-to-real gap could be large.
2. **Compute cost**: FDTD simulations at high resolution can be slow. Need to balance resolution with practicality.
3. **Overfitting to synthetic patterns**: If synthetic data dominates training, the model might learn simulation artifacts rather than real GPR physics.
4. **Dielectric uncertainty**: The permittivity values from the literature are approximate. Real fossils vary widely depending on mineralization, host rock, and moisture.

## Files to create

- `src/data/generate_gprmax_models.py` - Script to generate gprMax input files
- `src/data/process_gprmax_output.py` - Convert gprMax output to training images
- `configs/synthetic_data.yaml` - Parameters for synthetic data generation
- `notebooks/04_synthetic_data_exploration.ipynb` - Visualize and validate synthetic data
