# Reading Notes: GPR Vertebrate Skeleton Detection (2024)

**Citation**: Peredo et al. (2024). Detection of vertebrate skeletons using GPR trace analysis. Ica Desert, Peru.
**Read**: 2026-03-23

---

## What this paper is about

Uses GPR to detect a buried whale skeleton at the Ica Desert paleontological site in Peru. This isn't ML-based detection. It uses physics-based trace analysis: forward modeling with Ricker wavelets, reflectivity analysis, and polarity pattern matching to distinguish bone from surrounding sediment.

## Key technical details

**Equipment**: 400 MHz GSSI antenna. This is a standard mid-frequency GPR setup. The 400 MHz frequency gives reasonable resolution for targets at shallow depths (0-2m).

**Forward modeling approach**: They simulate what a GPR trace should look like when it encounters bone vs. sediment. The model uses known dielectric properties:
- Bone (fossilized): permittivity ~7-12 depending on mineralization
- Dry sand/sediment: permittivity ~3-5
- The contrast between bone and sediment produces a detectable reflection

**The polarity triplet signature**: This is the most useful finding for our project. When a GPR pulse hits a buried bone, the reflected trace shows a characteristic positive-negative-positive polarity pattern. This is because:
1. The bone has higher permittivity than the surrounding sand
2. The pulse reflects off the top surface (positive)
3. Then off the internal structure or bottom surface (negative, phase-reversed)
4. Then a trailing positive

They use this triplet as a diagnostic signature to distinguish bone from other buried objects.

**Ricker wavelet**: The source pulse is modeled as a Ricker wavelet (Mexican hat wavelet), which is standard in GPR modeling. The central frequency determines the wavelet shape.

**Reflectivity plots**: They compute reflectivity coefficients from the dielectric contrasts and overlay them on the B-scan data. Locations where the observed reflectivity matches the expected bone signature are flagged.

## What I want to steal from this paper

- **The polarity triplet idea.** If we're generating synthetic GPR data for fossil targets, the synthetic traces should reproduce this positive-negative-positive pattern. This gives us a physics-based constraint for synthetic data generation.
- **Dielectric property values for bone.** We need realistic permittivity values for mineralized bone in different sediment matrices. This paper provides actual numbers.
- **The trace-level analysis approach.** Before jumping to image-level CNN detection, we should understand what the raw trace signatures look like. This could inform feature engineering or at least help us validate that our model is learning real physics.
- **Forward modeling as a sanity check.** Even if we use data-driven detection, we should model what we expect to see before looking at real data.

## Limitations

- Single site, single specimen. The whale skeleton is large and shallow. Smaller fossils at greater depths would produce much weaker signatures.
- No ML component at all. The detection is done by expert analysis of trace patterns. The question is whether ML can automate this.
- 400 MHz antenna may not be optimal for all fossil sizes. Smaller targets need higher frequencies.
- Fossilized whale bone has different material properties than, say, a dinosaur tooth or a small reptile skeleton. The permittivity values are specific to this specimen.

## Relevance to our project: VERY HIGH

This is the closest published work to what we want to do. The physics they describe is exactly what our ML model needs to learn from synthetic data. The polarity triplet signature gives us a concrete target pattern for synthetic data generation with gprMax.

## 3-2-1

**3 things I learned:**
1. Fossilized bone produces a positive-negative-positive polarity triplet in GPR traces due to the permittivity contrast with surrounding sediment.
2. The dielectric permittivity of mineralized bone ranges from ~7-12, while dry sediment is ~3-5. This contrast is detectable but not huge.
3. Forward modeling with Ricker wavelets can predict what a bone signature should look like before seeing real data.

**2 things I still don't know:**
1. How much the polarity triplet degrades with depth, noise, and different sediment types (wet clay vs. dry sand vs. gravel).
2. Whether smaller fossils (sub-10cm) produce detectable signatures at frequencies practical for field surveys (200-800 MHz).

**1 thing I want to investigate:**
Generate synthetic GPR traces in gprMax using the dielectric values from this paper and verify that the polarity triplet signature appears in the modeled data.
