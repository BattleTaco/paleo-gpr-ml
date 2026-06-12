---
name: reference-cuda-machine-setup
description: "Michael's CUDA machine is an RTX 5090 (Blackwell) that needs nightly cu128 PyTorch; how the synthetic step runs there"
metadata: 
  node_type: memory
  type: reference
  originSessionId: e6c235d4-3a76-4721-a696-dfcdc468b126
---

Michael's personal training machine has an **RTX 5090 (Blackwell, compute capability sm_120)**. Blackwell needs **CUDA 12.8+ and a PyTorch built against cu128**; older/stable wheels may not run on it, so use the **nightly cu128** build.

Install order on that machine (full guide in `docs/notes/12_cuda_runbook.md`):
1. `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`
2. `pip install -r requirements-cuda.txt` (project deps; intentionally does NOT pin torch)
3. `pip install -e . --no-deps`
Verify: `torch.cuda.get_device_capability(0)` should be `(12, 0)`.

`requirements.txt` is an old exact freeze from that box (cu128 nightly, pins expire). `requirements-cuda.txt` is the canonical forward-compatible install. The Mac uses `environment.yml` (CPU/MPS) and is for dev only.

The synthetic data step is staged to run there: `notebooks/05_synthetic_data_generation.ipynb` (GPU check → generate controlled models → run gprMax → fidelity gate → convert to images+YOLO labels). gprMax is built from source on that machine (not on PyPI); GPU FDTD needs pycuda + the CUDA 12.8 toolkit. The converter is `src/data/process_gprmax_output.py`. Do not train on synthetic data until the fidelity gate passes. See [[project-novel-gpr-fossil-research]], [[reference-docs-layout-and-obsidian]].
