# RESUME HERE

**Updated**: 2026-10-29 evening.
**One-line state**: main controlled set is DONE: 288/288 simulated, gate PASS 72/72, images +
labels written (bgr + raw). Next: experiment_03 C0-C2 in notebooks/06 (per-stratum reporting, A1.7).

## 2026-10-29 evening: pipeline refactor + main set

The pilot turned up two bugs that would have poisoned training. Both are fixed and recorded in
`docs/experiments/experiment_03_transfer.md` amendment A1:
- **Labels missed the targets.** The analytic box left out the Ricker delay (sqrt(2)/f, about
  750 samples at 400 MHz). 0 to 1% of target energy fell inside the 400 MHz boxes. Boxes now
  come from the measured target response, `bscan - null_twin`.
- **Gate probe read the wrong window** (the same missing delay). Replaced with a paired-polarity
  test: NCC(bone, anti_bone) << 0 and NCC(void, anti_bone) >> 0 in the top-reflection window.
  The pilot passes 8/8 under the new gate.

Pipeline (all config-driven, resumable, tested; 50 tests pass):
```bash
python -m src.data.generate_gprmax_models --config configs/synthetic_controlled.yaml
python -m src.data.run_gprmax_batch       --config configs/synthetic_controlled.yaml --workers 8
python -m src.data.synthetic_fidelity     --config configs/synthetic_controlled.yaml   # exits 1 on FAIL
python -m src.data.process_gprmax_output  --config configs/synthetic_controlled.yaml
```
- Data: `data/processed/synthetic/controlled/` (main), `.../controlled_pilot/` (pilot, moved).
  Images: `<data dir>/images/{bgr,raw}/<target_type>/{images,labels}`. Labels are identical
  across variants; `bgr` (background-removed) is the primary input.
- Throughput on the 5090: 36 s/model with 1 worker, 13 s at 4, 11 s at 8 (GPU close to saturated).
  The full 288 takes about 50 min.
- Geometry views are off (they were 80% of pilot disk). Logs: `<data dir>/logs/`, `run_log.jsonl`.
- Notebook 05 is now a thin driver over these modules, with outputs cleared.

Open items:
1. DONE 2026-10-29: gate PASS 72/72, 288 images x 2 variants. Limestone 800 MHz bone TCR is
   median 1.65 (min 0.75), so report per host x freq. Load manifests with
   `process_gprmax_output.load_manifest` (pandas otherwise reads "null" as NaN).
2. Write `notebooks/06_transfer_experiment.ipynb`. Split by scene cell (A1.6). Report C2 per
   frequency (A1.2: at 400 MHz bone is a mixed-polarity doublet).
3. The `research-integrity-check` skill before any number is reported.
4. Not yet covered: wet/clay hosts (anti_bone is undefined when host eps > bone eps), 2 GHz.

## History

### 2026-10-29 update: GPU driver broke on kernel bump

- Machine had moved to kernel `7.0.0-28` with no nvidia module for it (HWE meta lagged), so
  `cuInit` failed. The 7.0.0-28 module package pins driver 580.173.02, which Ubuntu withdrew.
- Fix applied: install `linux-generic-hwe-24.04 linux-modules-nvidia-580-open-generic-hwe-24.04
  nvidia-driver-580-open` (kernel 7.0.0-42 + driver 580.178.04, signed prebuilt module; Secure
  Boot is on, so avoid DKMS), then reboot into 7.0.0-42.
- Notebook 05 `_gpu_available()` now calls `cuda.init()` + checks device count. Before, it only
  imported pycuda, which succeeds with no driver, so the "CPU fallback" would have crashed.
- Batch still 17/32. Next: verify `nvidia-smi` + pycuda on the new driver, then resume step 2 below.

### 2026-10-29 later: batch done, polarity holds on all 8 scenes

- Driver fix verified (7.0.0-42, 580.178.04, pycuda + torch see the 5090). Remaining 15 models
  ran on GPU (~37s each). 32/32 merged, `FIDELITY GATE: PASS`, 32 PNG + YOLO labels written to
  `data/processed/synthetic/images/` (null labels empty, as intended).
- Ran the gate's probe on every scene (resume steps 3 and 4). Bone -1, void +1, anti_bone +1 in
  all 8 (both hosts, both depths, both freqs). Center-column picks except one, noted below.
  void/bone |amp| ~1.4-1.7 in dry sand, ~3.0-3.5 in limestone. bone and anti_bone amplitudes
  match closely in limestone (0.0217 vs 0.0217 at d010 f0400), so the per-host anti-bone
  magnitude matching works.
- Weak spots: the 20 cm / 400 MHz scenes have bone/null amplitude of only ~4 (vs 30-70 elsewhere).
  In `limestone_d020_f0400` the strongest bone/anti_bone return is at the edge column (col 0),
  not the apex. The center-column sign is still correct.
- **Training PNGs have no background removal.** `process_gprmax_output` goes raw -> time_gain
  -> 99th-percentile clip, so the direct wave sets the clip level and squashes the target. In
  `limestone_d020_f0400__bone.png` the hyperbola is barely visible. The gate removes background,
  the images do not. Decide which input C0-C2 uses (and which the real benchmark uses) before
  training, and record it in experiment_03 since the doc is pre-registered.
- **Set size is pilot scale.** experiment_03 section 3.2 calls for ~75 scenes per target type with
  clutter realizations and paired scene ids. We have 8 per type, no clutter, target always
  centered. Too small for 5 seeds + bootstrap CIs. Scaling `configs/synthetic_controlled.yaml`
  (x_center sweep, clutter, more depths / hosts) to ~300 models is ~3 h of GPU time.
- Notebook 05 now holds ~130k lines of gprMax log output. Clear outputs before committing.

### What is done and persistent (survives reboot)

The `paleo-gpr-ml` conda env is now GPU-ready on the RTX 5090 (Blackwell, sm_120):

- **torch** `2.12.0.dev+cu128`, sees the 5090 at compute capability `(12, 0)`.
- **gprMax 3.1.7** built from source at `~/opt/gprMax` (editable install, conda-forge gcc 14.3,
  built against the env's numpy 2.4.3). Moved out of `/tmp` so it persists. All 7 Cython
  extensions compiled and import.
- **GPU FDTD** works: installed CUDA 12.9 toolkit (`cuda-nvcc` etc., nvcc supports `compute_120`)
  plus `pycuda 2026.1`. `gprMax -gpu` runs on the 5090 and JIT-compiles kernels for sm_120.
- **project** installed editable (`pip install -e . --no-deps`) so `src` imports from any cwd.
- Build toolchain (`c-compiler`, `cxx-compiler`, `make`) is in the env via conda-forge.

Sanity check after reboot:
```bash
conda activate paleo-gpr-ml
python -c "import torch, gprMax, pycuda.autoinit, pycuda.driver as d; print(torch.cuda.get_device_name(0), d.Device(0).compute_capability())"
# expect: NVIDIA GeForce RTX 5090 (12, 0)
```

### Notebook 05 changes already applied

- `run_gprmax` auto-detects pycuda and uses `-gpu`, else falls back to the CPU solver. Cells no
  longer force `-gpu`.
- The fidelity gate now does standard GPR background removal (subtract the trace-mean per row)
  before reading target polarity. The raw direct wave was swamping the target reflection, which
  made the old probe read the same sign for every target.

### Data progress (persistent, in `data/processed/synthetic/controlled/`, gitignored)

- 32/32 `.in` files generated.
- 17/32 merged B-scans done. The remaining 15 still need to run.
- Loose part-files from the interrupted model were cleaned.

### The finding (SUPERSEDED 2026-10-29, see amendment A1.1)

> This probe read a window before the target arrival (missing Ricker delay). The sign pattern
> was consistent but came from the wavelet tail, not the main reflection. The paired-polarity
> gate in `src/data/synthetic_fidelity.py` replaces it. Do not cite the numbers below.

On scene `dry_sand_d010_f0400`, with background removal:
- bone top polarity **-1** (correct, high eps), void **+1** (correct, low eps), anti_bone **+1**.
- Apex centered (column 48 of 96). |void| ~1.37x |bone|, same direction as the 1D note's
  "bone ~1.7x weaker."
- So the controlled synthetic data reproduces the polarity inversion H1 rests on. The earlier
  gate FAIL was the analysis probe, not the physics. Confirm across more scenes before logging.

### Resume steps (in order)

1. Run the sanity check above.
2. Finish the batch and run the gate + conversion. Either:
   - Headless: `KMP_DUPLICATE_LIB_OK=TRUE python scripts/run_nb05.py`
     (runs the notebook with cwd pinned to repo root; needed because the notebook uses
     repo-root-relative paths and nbconvert/nbclient default the kernel cwd to `notebooks/`).
   - Or open `notebooks/05_synthetic_data_generation.ipynb` in Jupyter launched from the repo
     root and run top to bottom.
   The run skips the 17 already-merged models and runs the remaining 15 on the GPU (~10 min).
3. Confirm `FIDELITY GATE: PASS` in the output and eyeball the hyperbola images.
4. Spot-check polarity on a limestone scene and an 800 MHz scene (not just dry_sand 400 MHz).
5. If the gate holds, the controlled sets (void / bone / anti_bone / null) are ready. Move to
   `docs/experiments/experiment_03_transfer.md`, conditions C0-C2 in a new
   `notebooks/06_transfer_experiment.ipynb`:
   - C0a / C0b same-domain ceilings, C1 naive void->bone transfer (H1a), C2 bone vs anti_bone
     polarity control (H1b), against the polarity matched-filter baseline (Cbase).
   - Run the `research-integrity-check` before reporting any number.

### Notes / gotchas

- Do not reinstall torch from the generic specs; it would pull a non-Blackwell wheel and break
  CUDA. The cu128 nightly is the only one that works here.
- gprMax needs `nvcc` on PATH at runtime (it is, inside the activated env). If GPU ever fails,
  the notebook auto-falls back to CPU (~2 hours for the full 32 vs ~20 min on GPU).
- These 2D models are tiny, so the 5090 is not compute-saturated. The per-trace Python/JIT/IO is
  the bottleneck. Saturating the card would need the larger 3D models in the scale-up plan.
