# Controlled Synthetic Generator (gprMax input files)

**Date**: 2026-06-10
**Code**: `src/data/generate_gprmax_models.py`, `configs/synthetic_controlled.yaml`, `tests/test_generate_gprmax_models.py`
**Implements**: experiment_03 §3.2 (controlled synthetic sets) for H1.

---

## What I built and why

The detector-side experiment (H1) depends entirely on the synthetic comparison being clean. If bone and void scenes differ in a lot of ways, any transfer gap I measure could come from anything. So I built the generator around one rule from the pre-registered spec:

**Hold geometry, depth, host, target shape/position, and antenna constant across target types. Vary ONLY the target dielectric.**

For every grid cell `(host, depth, frequency, x_center)` it emits the *same* 2D scene four times, one per target type, sharing a `scene_id` so they pair exactly:

| type | eps | polarity | role |
|---|---|---|---|
| void | 1 | positive top | matches my real cavities (low-eps) |
| bone | 9 (7-12 range) | negative top | the fossil-like target |
| **anti_bone** | computed per host | positive top | **the H1b causal control** |
| null | = host | none | negative control, should be ~invisible |

### The anti-bone control (the important part)

`anti_bone` is a low-permittivity target whose **|reflection coefficient| equals bone's but with the opposite sign**. It isolates *polarity* from *contrast magnitude* and *appearance*: if a void-trained detector finds anti-bone but not bone, the only thing that changed is the sign -> polarity is the cause (H1b). I derived its permittivity in closed form (`matched_low_eps_target`): with `sh=sqrt(host)`, `m=|r_bone|`, then `eps_anti = (sh*(1-m)/(1+m))^2`. For dry sand (eps 4) and bone (eps 9): `r_bone=-0.2`, so `eps_anti≈1.78`, giving `r=+0.2`. Verified by test.

## Verification

- I confirmed the control invariant *directly*: the `bone` and `anti_bone` `.in` files for a given scene are **byte-identical except the target's `#material` permittivity** (9 vs 1.778). Same domain, cylinder, antenna, frequency, everything.
- `tests/test_generate_gprmax_models.py` (9 tests): anti-bone magnitude/sign matching, the worked numeric value, null==host, the per-scene "only eps varies" invariant, grid size, geometry/depth/trace math, required `.in` directives present, and end-to-end file+manifest writing. All pass (full suite 20/20).
- ruff + format + mypy clean.
- Ran it: 32 models written (2 hosts x 2 depths x 2 freqs x 4 targets) + `manifest.csv` with ground-truth target position/radius/depth per file (for generating detection labels later). Output under `data/processed/synthetic/controlled/` (gitignored, it's regenerable from the code + config).

## gprMax format choices (for future-me)

- 2D model: one cell thick in z (`#dx_dy_dz` equal, domain z = dx).
- `#box` fills the ground (y in [0, surface]); air gap above for the antenna.
- `#hertzian_dipole` (tx) + `#rx` on the surface; `#src_steps`/`#rx_steps` march them across the line -> run with `gprmax <file>.in -n <n_traces>` to assemble a B-scan.
- Ricker source at the scene frequency.
- `#cylinder` target at `(x_center, surface - depth)` with the scene radius.
- A `#geometry_view` is emitted so I can eyeball each model in Paraview before trusting it.

## Honest limitations / decisions

- **Not yet simulated.** gprMax isn't installed on the Mac (deferred, see note 07). The `.in` files are written and structurally tested, but the FDTD output is unverified until they run on the CUDA box. I'm explicit that "tests pass" means "the input files are well-formed," not "the physics output is validated." The §3.2 fidelity check (reproduce Peredo's triplet / Catanzariti's hyperbola) is the gate before any sim is used for training.
- **Idealized geometry.** Target is a cylinder (uniform cross-section), homogeneous host, no clutter yet, lossless target. That's deliberate for the *first* clean causal test. Shape variety, heterogeneous soil (fixed-seed `#fractal_box` so it's identical across target types), and conductivity sweeps come after H1a/H1b are decided, adding them now would muddy the control.
- **Small grid.** 32 models is a pilot to validate the pipeline. The full set scales the same config up (more depths/hosts/positions/seeds).

## Next steps

1. Move `.in` files to the CUDA box, install/build gprMax there, run a single model, and **validate fidelity** (note 08 physics: bone negative top reflection, void positive; reproduce a hyperbola). Gate before scaling.
2. `src/data/process_gprmax_output.py`, convert gprMax HDF5 B-scans to images + YOLO labels (from the manifest's known target geometry), matching the real-data format.
3. Scale the grid; add heterogeneous-soil variants with fixed per-scene seeds.
4. Then C0-C2 detector runs (experiment_03).
