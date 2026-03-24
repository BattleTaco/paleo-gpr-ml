# Reading Notes: GPR Dinosaur Bones Sicily (2023)

**Citation**: Catanzariti et al. (2023). GPR survey detecting theropod dinosaur bones in Cretaceous limestone cave, Sicily.
**Read**: 2026-03-23

---

## What this paper is about

GPR survey inside a cave in Sicily where theropod dinosaur bones were found in Cretaceous limestone. They use a 2 GHz antenna to map potential additional bone deposits in the cave walls and floor. The analysis relies on signal processing (Kirchhoff migration, Hilbert transform) rather than ML.

## Key technical details

**Equipment**: 2 GHz antenna. This is high-frequency GPR, which gives much better spatial resolution than the 400 MHz in the Ica Desert paper but at the cost of depth penetration. Good for shallow targets in rock, which is exactly the cave scenario.

**Kirchhoff migration**: A standard GPR processing technique that collapses diffraction hyperbolas back to point reflectors. This is important because raw B-scans show hyperbolic signatures from buried objects. Migration "focuses" these back to their true spatial locations. If we're doing detection on B-scans, we need to decide whether to detect on raw (pre-migration) or migrated data.

**Hilbert transform / instantaneous amplitude**: They compute the analytic signal using the Hilbert transform and extract the instantaneous amplitude (envelope). This is equivalent to computing the magnitude of the analytic signal. It highlights reflectors regardless of polarity and can make anomalies pop out more clearly. We actually have `instantaneous_amplitude` already implemented in our `features/build_features.py`.

**Hyperbolic reflections from bones**: The bones produce classic diffraction hyperbolas in the raw B-scan. The apex of each hyperbola marks the bone location. The shape of the hyperbola encodes depth and the electromagnetic wave velocity in the medium. This is very similar to what utilities produce in our infrastructure GPR dataset.

**Results**: Found 5 reflective areas consistent with additional dinosaur remains beyond the known excavation site. These were identified by combining migrated radargrams with amplitude analysis.

## What I want to steal from this paper

- **The parallel between bone hyperbolas and utility hyperbolas.** Both produce similar diffraction patterns. This means our infrastructure GPR model (trained on utilities) might partially transfer to fossil detection, at least for the geometric pattern recognition component.
- **Migration as a preprocessing choice.** We should experiment with both raw and migrated data for detection. Migration removes the hyperbolic shape, which could be either helpful (cleaner point targets) or harmful (removes a distinctive pattern the model could learn).
- **Hilbert transform features.** We already have this implemented. Should definitely include it as an input channel or feature for detection models.
- **High-frequency GPR for paleontology.** The 2 GHz frequency matters for thinking about what field equipment would be needed for fossil prospecting. Different frequency = different resolution = different signature appearance.

## Limitations

- Case study, not a systematic method. They found bones they already knew about and then confirmed them with GPR.
- No quantitative detection metrics. It's visual interpretation by experts.
- Limestone matrix is specific. The dielectric contrast between bone and limestone is different from bone and unconsolidated sediment.
- 2 GHz data looks very different from 400 MHz data. Models trained on one frequency may not transfer to another.

## Relevance to our project: VERY HIGH

Confirms that GPR can detect dinosaur bones in situ. The hyperbolic signature is similar to what our infrastructure dataset already contains (utility hyperbolas). This supports the idea that transfer from infrastructure GPR to fossil GPR might work.

## 3-2-1

**3 things I learned:**
1. At 2 GHz, dinosaur bones in limestone produce clear diffraction hyperbolas that look geometrically similar to utility pipe reflections in infrastructure GPR.
2. Kirchhoff migration is the standard way to collapse hyperbolas to point reflectors. Pre-migration vs. post-migration data are fundamentally different inputs for a detection model.
3. Instantaneous amplitude (Hilbert transform envelope) is a standard derived feature for highlighting GPR anomalies regardless of polarity.

**2 things I still don't know:**
1. How much the hyperbolic signature shape varies between different bone types, sizes, and orientations relative to the survey line.
2. Whether a model trained on 400 MHz infrastructure data can transfer to 2 GHz paleontological data (the frequency-dependent differences in wavelet shape and resolution are significant).

**1 thing I want to investigate:**
Compare the hyperbolic signatures in our infrastructure dataset (utility class) with published GPR signatures of fossils to quantify how similar they actually are. If they're geometrically similar, domain transfer has a better chance.
