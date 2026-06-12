# Environment Setup: Dedicated Conda Env for the Research

**Date**: 2026-06-10
**Status**: Done (env built + verified; gprMax deferred)

---

## Why I'm doing this now

I went to start the synthetic-data phase and realized I'd been running everything through my homebrew system Python (`/opt/homebrew/bin/python3`, 3.12), no scipy, no torch, no gprMax. That's a trap: an undocumented, drifting environment is the fastest way to make research irreproducible. Before I write another line of research code, I want one **dedicated conda environment** that is *the* environment for this paper. Everything installs there, every notebook runs there, and the spec lives in the repo so I (or anyone) can rebuild it exactly.

This is a foundation step, but it's the right one. Reproducibility is part of publishable rigor.

## The machine

- macOS, **Apple Silicon (arm64)**, miniconda3, conda 25.5.1.
- No CUDA here. PyTorch runs on **CPU / MPS** on this machine.
- The heavy training box (the one behind `requirements.txt`) is Linux + CUDA. That's a *separate* environment, `requirements.txt` is a CUDA/nightly freeze and is **not** meant to be installed on the Mac. This conda env is my local research/dev environment; the CUDA box is for heavy runs.

## What I set up

Env name: **`paleo-gpr-ml`** (matches the repo). Spec: `environment.yml` (rewrote it; the old one was generic and had stale pins like `gprpy`/`mlflow` I'm not using).

Layout decisions:
- **Core scientific stack from conda-forge** (numpy, scipy, pandas, scikit-learn/image, h5py, matplotlib, seaborn, jupyterlab). conda-forge builds are the cleanest on arm64.
- **Deep-learning + CV stack from pip** (torch, torchvision, torchmetrics, ultralytics, timm, albumentations, opencv, segmentation-models-pytorch, monai). On Mac the pip wheels give me MPS-capable torch with the least friction.
- **Build toolchain** (`c-compiler`, `llvm-openmp`, `cython`) included on purpose, gprMax needs a C compiler + OpenMP, so I want that ready.
- **Dev tooling** (pytest, ruff, mypy) to match `pyproject.toml`.

### gprMax is a separate, deliberate step

I left **gprMax out of the main env solve.** It needs a C compiler + OpenMP and is historically finicky on Apple Silicon, and a pip failure inside `conda env create` would tank the whole build. So I install the rest first, get a known-good env, then attempt gprMax on its own where I can see exactly what breaks. Worst case, gprMax doesn't build cleanly on the Mac, that's fine, because:
- The model *generator* I'm about to write only emits gprMax `.in` text files; it doesn't need gprMax installed to run.
- The actual FDTD simulations are compute-heavy and are better run on the CUDA box anyway (per `docs/planning/synthetic_data_plan.md`).

## The one snag: dual-OpenMP collision (and the fix)

First import of the ML stack aborted with:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

This is the classic Apple-Silicon conda+pip-torch problem: there are **three** OpenMP runtimes in the env (`lib/libiomp5.dylib`, conda `lib/libomp.dylib`, and torch's own `torch/lib/libomp.dylib`), and loading more than one aborts the process. The core scientific stack (numpy/scipy/sklearn/etc.) imported fine, it only bit when torch loaded.

**Fix**: set `KMP_DUPLICATE_LIB_OK=TRUE`, persisted on the env so it applies on every activation:

```bash
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE -n paleo-gpr-ml
```

Honest tradeoff: this is the widely-used, accepted workaround, but it's officially "unsafe/unsupported", it lets multiple OpenMP runtimes coexist instead of truly de-duplicating them. For my work (torch on MPS, numpy on its own BLAS) it's fine; I haven't seen it cause wrong results in this kind of pipeline. If I ever do heavy mixed CPU-OpenMP numerics and get suspicious results, revisit this (e.g. pin everything to a single OpenMP, or get torch from conda-forge so it shares the conda runtime). Noting it here so future-me remembers it's a known knob, not magic.

## How to rebuild / use

```bash
conda env create -f environment.yml      # build it
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE -n paleo-gpr-ml   # OpenMP fix
conda activate paleo-gpr-ml              # use it
python -m ipykernel install --user --name paleo-gpr-ml --display-name "Python (paleo-gpr-ml)"
```

`environment.lock.yml` is the fully-pinned export (`conda env export --no-builds`) for exact reproduction.

## Verification (2026-06-10)

- [x] `conda env create` completed without errors (exit 0). Python **3.11.15**.
- [x] Core: `numpy 2.4.6, scipy 1.17.1, pandas 3.0.3, sklearn 1.9.0, skimage 0.26.0, h5py 3.16.0, matplotlib 3.10.9, seaborn 0.13.2, PIL 12.2.0`, all import.
- [x] `torch 2.12.0`, imports clean (after OpenMP fix). **CUDA: False, MPS: True** (as expected on this Mac).
- [x] `torchvision 0.27.0, torchmetrics 1.9.0, ultralytics 8.4.64, timm 1.0.27, albumentations 2.0.8, cv2 4.13.0, segmentation_models_pytorch 0.5.0, monai 1.5.2, omegaconf 2.3.0`, all import.
- [x] Jupyter kernel registered: `Python (paleo-gpr-ml)`.
- [x] Lock file written: `environment.lock.yml`.
- [ ] **gprMax: DEFERRED.** Not on PyPI (`pip install gprMax` -> "No matching distribution"). It's source-built from GitHub (Cython + C compiler + OpenMP), which is a known rabbit hole on Apple Silicon. Per `docs/planning/synthetic_data_plan.md` the FDTD sims run on the CUDA box anyway, and the `.in`-file generator I'm about to write doesn't need gprMax installed. So I'm deferring it to a dedicated setup task (build from source on the CUDA box, or a focused Mac source-build attempt later). **This does not block the synthetic-data phase.**

## Next step after this

Once the env is verified, start the synthetic-data phase:
1. Cheap **1D forward model** (numpy) to validate the bone polarity-triplet physics before committing FDTD compute.
2. `src/data/generate_gprmax_models.py`, parameterized gprMax `.in` file generator grounded in the dielectric values from Peredo (2024) and Catanzariti (2023).

See `docs/planning/synthetic_data_plan.md` and `docs/planning/next_steps_todo.md`.
