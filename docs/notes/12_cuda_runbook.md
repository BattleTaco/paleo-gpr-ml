# CUDA Runbook (RTX 5090 / Blackwell)

**Date**: 2026-06-10
**Purpose**: Everything I need to do on my personal machine to run the synthetic data step, set
up in advance so I do not have to figure it out in the moment.

The machine has an RTX 5090, which is Blackwell (compute capability sm_120). Blackwell needs
CUDA 12.8 or newer and a PyTorch built against cu128. Older torch wheels will not run on it.

## 1. Python environment

I keep the env spec in `environment.yml` (Mac) and `requirements-cuda.txt` (this CUDA machine).
On the 5090, do NOT install torch from the generic specs, because they pull a non-Blackwell
wheel. Install torch first from the nightly cu128 index, then the rest.

```bash
conda create -n paleo-gpr-ml python=3.11 -y
conda activate paleo-gpr-ml

# 1) PyTorch nightly for Blackwell (cu128):
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# 2) the rest of the stack:
pip install -r requirements-cuda.txt

# 3) the project package, without re-resolving torch:
pip install -e . --no-deps

# 4) jupyter kernel:
python -m ipykernel install --user --name paleo-gpr-ml --display-name "Python (paleo-gpr-ml)"
```

Check the GPU is actually seen and is Blackwell:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
```

I want a 2.x.dev version, "NVIDIA GeForce RTX 5090", and capability `(12, 0)`. If
`torch.cuda.is_available()` is False or the capability is wrong, the wrong wheel got installed,
so redo step 1.

## 2. gprMax

gprMax is not on PyPI. Build it from source.

```bash
git clone https://github.com/gprMax/gprMax.git
cd gprMax
pip install -r requirements.txt   # gprMax's own deps (cython, etc.)
python setup.py build
pip install -e .
```

CPU works out of the box. For GPU-accelerated FDTD (much faster, and the reason to be on this
machine) gprMax uses pycuda, which needs the CUDA 12.8 toolkit installed:

```bash
pip install pycuda
# then run models with the -gpu flag (see the notebook)
```

If pycuda or the CUDA toolkit fights me on Blackwell, the CPU solver still produces correct
B-scans, just slower. The science does not change, only the runtime.

## 3. Run the work

Open `notebooks/05_synthetic_data_generation.ipynb` with the `paleo-gpr-ml` kernel and run it
top to bottom. It is written to explain each step, what to expect, and what a good result looks
like. In short it does:

1. Check torch sees the 5090 and gprMax imports.
2. Generate the controlled `.in` files (`src/data/generate_gprmax_models.py`).
3. Run gprMax on them and merge each B-scan.
4. The fidelity gate: confirm bone reflects negative at the top and void positive, and that a
   hyperbola appears. This matches the 1D physics in `docs/notes/08_forward_model_1d_validation.md`.
   If this fails, stop and fix the model before going further.
5. Convert the B-scans to images + YOLO labels (`src/data/process_gprmax_output.py`).
6. Eyeball the dataset.

## 4. Where the outputs go

- `.in` files and merged `.out` files: `data/processed/synthetic/controlled/` (gitignored).
- Images + YOLO labels: `data/processed/synthetic/images/<target_type>/`.
- All of `data/` is gitignored and regenerable from the code, so I do not commit it. The code,
  config, and this runbook are what I version.

## 5. The gate before training

Do not train any detector on synthetic data until the fidelity check in step 4 passes. The
whole hypothesis (`docs/notes/09_research_hypothesis.md`) rests on the synthetic physics being
right, so this is the one checkpoint I do not skip.
