# RESUME HERE

**Paused**: 2026-06-11 evening (machine going to sleep mid synthetic-data run).
**One-line state**: GPU pipeline is built and working; the controlled synthetic batch is 17/32
done; the fidelity gate is verified to pass with the corrected probe. Resume by finishing the
batch, then move to the transfer experiment.

## What is done and persistent (survives reboot)

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

## Notebook 05 changes already applied

- `run_gprmax` auto-detects pycuda and uses `-gpu`, else falls back to the CPU solver. Cells no
  longer force `-gpu`.
- The fidelity gate now does standard GPR background removal (subtract the trace-mean per row)
  before reading target polarity. The raw direct wave was swamping the target reflection, which
  made the old probe read the same sign for every target.

## Data progress (persistent, in `data/processed/synthetic/controlled/`, gitignored)

- 32/32 `.in` files generated.
- 17/32 merged B-scans done. The remaining 15 still need to run.
- Loose part-files from the interrupted model were cleaned.

## The finding (verified, not yet logged)

On scene `dry_sand_d010_f0400`, with background removal:
- bone top polarity **-1** (correct, high eps), void **+1** (correct, low eps), anti_bone **+1**.
- Apex centered (column 48 of 96). |void| ~1.37x |bone|, same direction as the 1D note's
  "bone ~1.7x weaker."
- So the controlled synthetic data reproduces the polarity inversion H1 rests on. The earlier
  gate FAIL was the analysis probe, not the physics. Confirm across more scenes before logging.

## Resume steps (in order)

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

## Notes / gotchas

- Do not reinstall torch from the generic specs; it would pull a non-Blackwell wheel and break
  CUDA. The cu128 nightly is the only one that works here.
- gprMax needs `nvcc` on PATH at runtime (it is, inside the activated env). If GPU ever fails,
  the notebook auto-falls back to CPU (~2 hours for the full 32 vs ~20 min on GPU).
- These 2D models are tiny, so the 5090 is not compute-saturated. The per-trace Python/JIT/IO is
  the bottleneck. Saturating the card would need the larger 3D models in the scale-up plan.
