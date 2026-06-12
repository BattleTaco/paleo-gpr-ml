# 1D Forward-Model Physics Validation

**Date**: 2026-06-10
**Code**: `src/data/forward_model_1d.py` (library), `src/data/run_forward_model_validation.py` (experiment), `tests/test_forward_model_1d.py` (physics tests)
**Figure**: `results/figures/forward_model_1d_polarity.png`

---

## What I set out to do

Before I spend real compute generating FDTD (gprMax) B-scans of fossil-like targets, I wanted to confirm I actually understand the physics my papers describe, not just take Peredo's "polarity triplet" on faith. A full FDTD sim is heavy; a 1D convolutional forward model runs in milliseconds and lets me check the core reflection behaviour first. This is a de-risking step: if my mental model of the physics is wrong, I'd rather find out now than after 475 simulations.

Concretely, three questions:
1. Does a buried bone produce the kind of reflection signature Peredo et al. (2024) describe?
2. Are bone (high permittivity) and an air cavity (low permittivity) **polarity-inverted**? This matters a lot for me, my real dataset's anomalies are *cavities*, not bone.
3. How does the signature change with depth?

## How the model works

Standard 1D convolutional model, same one used in seismic/GPR: `trace(t) = wavelet(t) * reflectivity(t)`.

- A buried target is a layer with permittivity different from the host.
- At each interface the wave reflects with `r = (sqrt(eps_above) - sqrt(eps_below)) / (sqrt(eps_above) + sqrt(eps_below))` (normal incidence, non-magnetic).
- Each interface gets a spike of height `r` placed at its two-way travel time, and the whole reflectivity series is convolved with a Ricker wavelet (the standard GPR pulse).

Dielectric values are straight from the reading notes: bone ~7-12 (used 9), dry sand ~3-5 (used 4), air void = 1, limestone ~4-8. See `papers/reading_notes/02_*` and `03_*`.

**This is deliberately 1D.** It models reflection amplitude and polarity down a single column. It does **not** produce diffraction hyperbolas, those need 2D geometry, which is exactly what gprMax is for. So this validates the *reflection physics*, not the *spatial signature*.

## What I found

Reflection coefficients (dry sand host), and the peak polarity that actually shows up in the synthesized trace:

| Target | r (top, sand->target) | r (bottom, target->sand) | Trace polarity (top / bottom) |
|---|---|---|---|
| Bone (eps=9) | **-0.200** | +0.200 | **- / +** |
| Air cavity (eps=1) | **+0.333** | -0.333 | **+ / -** |
| Rock (eps=4.3, ~no contrast) | -0.018 | +0.018 | nearly invisible |

All automated verdicts PASS. Three takeaways:

1. **Bone and cavity are polarity-inverted.** Bone (higher eps than host) reflects *negative* at its top surface; an air void (lower eps) reflects *positive*. This is the single most important result for my research. It means a detector trained to find cavities is **not** automatically a bone detector, if it has learned anything polarity-sensitive, bone looks like the *opposite* of what it was trained on. This is a concrete, testable hypothesis for the transfer phase, and it's exactly the kind of domain-gap insight the paper needs.

2. **The cavity contrast is ~1.7x stronger than bone** (|0.333| vs |0.200|). Air-in-sand is a bigger dielectric jump than bone-in-sand. So fossil reflections are inherently weaker than the void reflections in my training data, another reason naive transfer could struggle. Bone in *limestone* (eps ~6) would be weaker still; bone in *wet* sediment could flip in either direction depending on moisture. Worth sweeping later.

3. **Velocity pull-up / push-down falls out naturally.** In the figure, the bone's bottom reflection arrives *later* (~16 ns) than the cavity's (~11 ns) even though both slabs are the same thickness at the same depth, because the wave slows inside high-eps bone and speeds up inside the air void. That's real GPR physics emerging from the model, which gives me confidence the model is behaving correctly.

The figure (`forward_model_1d_polarity.png`) has three panels: bone vs cavity vs rock over the same column; the bone/cavity polarity inversion zoomed in; and the bone signature at three depths (same shape, shifts later with depth, as it should).

## Honest limitations

- **1D only.** No hyperbolas, no antenna radiation pattern, no 2D scattering. The geometric signature that my CNN/YOLO actually keys on is not tested here. gprMax is still required.
- **Thin-bed tuning.** My first run used a 10 cm slab; at 400 MHz (wavelength ~0.37 m in dry sand) the top and bottom reflections overlapped and the per-interface polarity readout was misleading. I bumped the slab to 40 cm so the interfaces resolve. Real fossils are often *thinner* than this and will sit below tuning thickness, where the two reflections merge into one composite wiggle. That's a real resolution limit, not a bug, noted so I model it honestly later (and a reason higher-frequency antennas like Catanzariti's 2 GHz matter for smaller targets).
- **Single permittivity per material.** Real bone permittivity varies with mineralization and moisture; real soil is heterogeneous. These are point estimates from the literature, not measured values.
- **No noise/clutter.** Real GPR has system noise and ground clutter that I haven't added.

## How I verified it (by the book)

- `tests/test_forward_model_1d.py`: 11 physics-invariant tests (reflection coefficient bounds + signs + antisymmetry, the bone/cavity polarity inversion, monotonic travel times, Ricker shape/symmetry/zero-mean, energy response). All pass.
- One test (`test_ricker_wavelet_shape`) caught a real bug: my first wavelet used `np.arange(-L/2, L/2, dt)`, which drops the endpoint and left the pulse slightly off-center / non-symmetric. Fixed to a symmetric odd-length grid. Good example of why the tests exist.
- `ruff` clean, `ruff format` clean, `mypy` clean on all three files.
- Everything run in the `paleo-gpr-ml` conda env.

## Decisions

- The polarity-inversion result becomes an **explicit hypothesis to test in the transfer phase**: does a cavity-trained detector transfer to (polarity-inverted, weaker) bone targets, or does the inversion break it? Either answer is paper-worthy.
- Proceed to building the gprMax model generator. The dielectric values and scene geometry I validated here carry straight over.

## Open questions / next steps

1. **Build `src/data/generate_gprmax_models.py`**, parameterized gprMax `.in` generator (target type, depth, host medium, frequency), reusing these dielectric values. This is the next concrete step.
2. Sweep bone-in-limestone and bone-in-wet-sediment to see how much the contrast (and even the polarity) shifts with host medium and moisture.
3. When gprMax B-scans exist, check whether the 2D bone hyperbola is geometrically similar to my utility hyperbolas (Catanzariti's note suggests it should be), that's the *other* half of the transfer question alongside polarity.
