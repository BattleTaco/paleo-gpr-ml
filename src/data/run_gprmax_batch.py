"""Run every model in a generated manifest through gprMax and merge each B-scan.

Why a module instead of the notebook loop
-----------------------------------------
- Resumable by content, not by file existence: a merged file only counts as done if it opens and
  has the expected number of traces, so a run killed mid-merge is redone instead of trusted.
- Stale per-trace part files from an interrupted model are removed before rerunning it.
- gprMax's verbose output goes to one log per model, not into the notebook (the pilot run put
  ~130k lines of log into notebook 05).
- Parallel workers. These 2D models are small, so one gprMax process leaves the GPU mostly idle;
  the per-trace Python/JIT/IO overhead is the bottleneck. Several processes share the card.
- Stops after N consecutive failures, which is what a lost GPU driver looks like.

Run:
    python -m src.data.run_gprmax_batch --config configs/synthetic_controlled.yaml --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def gpu_available() -> bool:
    """True only if a CUDA device actually initializes.

    Importing pycuda is not enough: it imports fine when the kernel module is missing (e.g. after
    a kernel update), and gprMax -gpu then dies at cuInit.
    """
    try:
        import pycuda.driver as cuda

        cuda.init()
        return cuda.Device.count() > 0
    except Exception:
        return False


def merged_is_valid(path: Path, n_traces: int) -> bool:
    """A merged B-scan counts as done only if it opens and holds every trace."""
    if not path.exists():
        return False
    try:
        import h5py

        with h5py.File(path, "r") as f:
            shape = f["rxs"]["rx1"]["Ez"].shape
        return len(shape) == 2 and shape[1] == n_traces
    except Exception:
        return False


def clean_partials(raw_dir: Path, stem: str) -> int:
    """Remove per-trace outputs (<stem>1.out, <stem>2.out, ...) and a broken merged file."""
    removed = 0
    for p in raw_dir.glob(f"{stem}[0-9]*.out"):
        p.unlink()
        removed += 1
    merged = raw_dir / f"{stem}_merged.out"
    if merged.exists():
        merged.unlink()
        removed += 1
    return removed


def run_one(raw_dir: Path, row: dict, use_gpu: bool, log_dir: Path, env: dict) -> dict:
    """Simulate and merge one model. Returns a status record; never raises."""
    stem = Path(row["in_file"]).stem
    n = int(row["n_traces"])
    t0 = time.time()
    clean_partials(raw_dir, stem)

    sim = [sys.executable, "-m", "gprMax", str(raw_dir / row["in_file"]), "-n", str(n)]
    if use_gpu:
        sim.append("-gpu")
    merge = [sys.executable, "-m", "tools.outputfiles_merge", str(raw_dir / stem), "--remove-files"]

    log_path = log_dir / f"{stem}.log"
    with log_path.open("w") as log:
        for cmd in (sim, merge):
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            if proc.returncode != 0:
                return _status(
                    stem, "failed", t0, f"{cmd[2]} exit {proc.returncode}, see {log_path}"
                )

    if not merged_is_valid(raw_dir / f"{stem}_merged.out", n):
        return _status(stem, "failed", t0, f"merged file invalid, see {log_path}")
    return _status(stem, "ok", t0)


def _status(stem: str, status: str, t0: float, error: str = "") -> dict:
    return {"stem": stem, "status": status, "seconds": round(time.time() - t0, 1), "error": error}


def run_batch(
    raw_dir: Path,
    workers: int = 1,
    use_gpu: bool | None = None,
    max_consecutive_failures: int = 3,
    limit: int | None = None,
) -> list[dict]:
    """Run all pending models in raw_dir/manifest.csv. Returns the status records of this run."""
    rows = list(csv.DictReader((raw_dir / "manifest.csv").open()))
    pending = [
        r
        for r in rows
        if not merged_is_valid(
            raw_dir / f"{Path(r['in_file']).stem}_merged.out", int(r["n_traces"])
        )
    ]
    n_done = len(rows) - len(pending)
    if limit is not None:
        pending = pending[:limit]
    if use_gpu is None:
        use_gpu = gpu_available()

    log_dir = raw_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    run_log = raw_dir / "run_log.jsonl"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}

    print(
        f"[run_gprmax_batch] {n_done}/{len(rows)} already done, "
        f"{len(pending)} to run, solver={'GPU' if use_gpu else 'CPU'}, workers={workers}",
        flush=True,
    )
    if not pending:
        return []

    results: list[dict] = []
    lock = threading.Lock()
    stop = threading.Event()
    consecutive = 0
    t_start = time.time()

    def task(row: dict) -> dict:
        if stop.is_set():
            return _status(Path(row["in_file"]).stem, "skipped", time.time(), "batch aborted")
        return run_one(raw_dir, row, use_gpu, log_dir, env)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(task, r) for r in pending]
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                results.append(rec)
                if rec["status"] != "skipped":
                    with run_log.open("a") as fh:
                        fh.write(json.dumps({**rec, "time": time.strftime("%FT%T")}) + "\n")
                consecutive = consecutive + 1 if rec["status"] == "failed" else 0
                if consecutive >= max_consecutive_failures and not stop.is_set():
                    stop.set()
                    print(
                        f"[run_gprmax_batch] {consecutive} failures in a row, aborting", flush=True
                    )
                done = sum(r["status"] == "ok" for r in results)
                elapsed = time.time() - t_start
                eta = elapsed / max(len(results), 1) * (len(pending) - len(results))
                print(
                    f"[{len(results)}/{len(pending)}] {rec['stem']} {rec['status']} "
                    f"{rec['seconds']}s (ok={done}, eta {eta / 60:.0f} min) {rec['error']}",
                    flush=True,
                )
    return results


def main() -> None:
    from src.data.generate_gprmax_models import default_out_dir, load_config

    p = argparse.ArgumentParser(description="Run + merge gprMax models from a manifest.")
    p.add_argument("--config", type=Path, default=Path("configs/synthetic_controlled.yaml"))
    p.add_argument("--raw-dir", type=Path, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--cpu", action="store_true", help="Force the CPU solver")
    p.add_argument("--limit", type=int, default=None, help="Run at most N pending models")
    args = p.parse_args()

    raw_dir = args.raw_dir or default_out_dir(load_config(args.config))
    results = run_batch(
        raw_dir, args.workers, use_gpu=False if args.cpu else None, limit=args.limit
    )
    failed = [r for r in results if r["status"] != "ok"]
    print(f"[run_gprmax_batch] done: {len(results) - len(failed)} ok, {len(failed)} not ok")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
