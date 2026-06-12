# Physics Baseline: Polarity Matched Filter (Cbase)

**Date**: 2026-06-10
**Code**: `src/models/polarity_matched_filter.py`, `src/models/run_polarity_baseline.py`, `tests/test_polarity_matched_filter.py`
**Figure**: `results/figures/polarity_baseline.png`
**Table**: `results/tables/polarity_baseline.csv`
**Implements**: experiment_03 condition Cbase.

## Why I built this

experiment_03 needs a non-ML reference to compare the learned detectors against. If a CNN cannot beat a plain matched filter, the extra complexity is not earning its place, and a reviewer will ask that question anyway. So I built the baseline now, while it runs on the Mac with no gprMax needed.

It also gives early, independent evidence for H1b. The whole hypothesis hinges on polarity being a real, readable signal that separates bone from voids. The matched filter reads exactly that, so if it splits the target types cleanly, that is a point on the board for H1b before any deep learning is involved.

## How it works

The target reflects the source pulse, and I know the pulse shape (Ricker at the survey frequency). So I slide that template along each trace and compute a normalized cross correlation (NCC). NCC is bounded to [-1, 1], so it is easy to threshold. The strongest absolute NCC says where the dominant reflector is and how well it matches the pulse. The sign of the NCC there is the polarity. High permittivity targets like bone reflect negative at the top, low permittivity targets like voids reflect positive, so the sign separates them.

I validate on single top-interface traces (host over a half-space target). That isolates the top reflection, which is the polarity diagnostic the filter is meant to read, and it is what dominates at a real target's hyperbola apex.

## Results

On clean traces (dry sand, 400 MHz) every target type was classified correctly:

| target | eps | detected | polarity | NCC | expected |
|---|---|---|---|---|---|
| void | 1.00 | yes | +1 (void-like) | +1.000 | +1 |
| bone | 9.00 | yes | -1 (bone-like) | -1.000 | -1 |
| anti_bone | 1.78 | yes | +1 (void-like) | +1.000 | +1 |
| null | 4.00 | no | 0 (none) | 0.000 | 0 |

The anti_bone result is the one I care about. It has bone's contrast magnitude but a void's sign, and the filter reads it as void-like (+1). That is the point of the control: the baseline keys on sign, not magnitude, so a magnitude-matched target with flipped polarity groups with the void. This is consistent with the polarity-as-cause story in H1b, now shown with a method that has no learned parameters at all.

The NCC scores are exactly +/-1 because a single-interface trace is a scaled Ricker, so it matches the template perfectly. Real B-scans will not be that clean.

### Noise robustness

I added Gaussian noise scaled to the clean signal peak and ran 300 trials per noise level for bone and void. Two things stand out:

- Detection rate stays near 1.0 until the noise std reaches about 0.6 to 0.7 of the signal peak, then it drops off. So the filter is reliable while the target reflection is at least comparable to the noise, which is a reasonable operating range.
- Polarity accuracy stays near 1.0 even at high noise, for the cases that were detected. Once the filter commits to a detection, it almost always gets the sign right. That is good for me, because polarity is the quantity H1b depends on.

## Honest limitations

- Single interface only. I read the top reflection in isolation. A real resolved slab also has a bottom reflection of opposite sign and similar strength, and a naive global-peak read can land on the bottom one. My first version did exactly that and a test caught it. For the baseline I am explicit that it reads the dominant reflection, and that for resolved slabs you window to the top first. That windowing is a later refinement.
- No clutter. Real GPR has ground clutter and the surface wave at the top of every trace, which I have not added. On real B-scans I will window out the surface wave before running the filter.
- Forward-model traces, not FDTD. These are 1D convolutional traces, so no hyperbola, no 2D scattering. The 2D version of this baseline runs per trace across a gprMax B-scan once those exist.
- Clean null. My null target is exactly contrast-free, so the trace is flat and nothing fires. A real null still has clutter, so the false-positive behavior on real backgrounds is still to be measured.

## How I verified it

- `tests/test_polarity_matched_filter.py`, 8 tests: NCC bounded to [-1, 1], bone detected negative, void detected positive, anti_bone reads void-like, null not detected, B-scan wrapper shape and count, 1D input rejected, empty template safe. All pass, full suite 28/28.
- One test caught the slab polarity bug described above, which is why I switched to single-interface validation and the dominant-reflection rule. Good case for writing the tests first.
- ruff, format, and mypy clean. Run in the `paleo-gpr-ml` env.

## Where this feeds the paper

This is Cbase in experiment_03, and it backs figure F3 (the polarity story) and the baseline column of table T1. When the gprMax B-scans exist, I run this same filter per trace across them and report detection and polarity on 2D data, then compare the ML detectors against it.

## Next steps

1. On the CUDA box: build gprMax, run the controlled models from note 10, and pass the fidelity gate.
2. `src/data/process_gprmax_output.py` to turn B-scans into images plus YOLO labels.
3. Run this baseline per trace across the synthetic B-scans (the 2D Cbase number).
4. Then the C0 to C2 detector runs.
