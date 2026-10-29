"""Fidelity gate for the controlled synthetic set: is the polarity contrast H1 rests on present?

What it tests, and why this way
-------------------------------
H1 says a detector trained on low-eps targets (voids) transfers poorly to high-eps targets
(bone) because their top reflections have opposite polarity. The simulation must reproduce that
contrast or nothing downstream means anything.

Reading an *absolute* polarity is fragile in FDTD. The received pulse from a 2D dipole is not a
clean Ricker, and at 400 MHz the 4 cm antenna offset is deep in the near field, so the direct
wave is not a usable reference template (tested on the pilot: it gave void and bone the same
sign). Instead the gate uses the paired design, which needs no template:

  1. Target response = bscan(target) - bscan(null twin). Exact, because the twins differ only
     in the target.
  2. At the apex trace, take a window from just before the earliest target arrival to one
     wavelet period after it, i.e. the top reflection before the bottom reflection interferes.
  3. bone and anti_bone have equal |r| and opposite sign, so their responses there must be
     anti-correlated (NCC << 0) with matched amplitude. void and anti_bone are both low-eps, so
     they must be positively correlated.

It also reports target-to-clutter: response peak vs the 99th percentile of the background-removed
null twin, i.e. how visible the target is above the scene's own clutter.

Run:
    python -m src.data.synthetic_fidelity --config configs/synthetic_controlled.yaml
Exits non-zero if any complete scene fails, so it can gate a pipeline.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.data.preprocess_gpr import remove_background
from src.data.process_gprmax_output import (
    apex_trace_index,
    group_by_scene,
    merged_path,
    read_merged_bscan,
    target_response,
)

DEFAULT_GATE = {
    "max_ncc_bone_anti": -0.5,
    "min_ncc_void_anti": 0.5,
    "amp_ratio_range": [0.5, 2.0],
    "onset_fraction": 0.1,
}


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-lag normalized correlation of two equal-length signals, in [-1, 1]."""
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a @ a) * (b @ b)))
    return float(a @ b) / den if den > 0 else 0.0


def onset_sample(trace: np.ndarray, fraction: float) -> int:
    """First sample where |trace| reaches ``fraction`` of its peak."""
    env = np.abs(trace)
    return int(np.argmax(env >= fraction * env.max()))


def top_window(onset: int, period: int, n: int) -> slice:
    """From a quarter period before the earliest arrival to one period after it."""
    return slice(max(0, onset - period // 4), min(n, onset + period))


@dataclass(frozen=True)
class SceneFidelity:
    scene_id: str
    ncc_bone_anti: float
    ncc_void_anti: float
    amp_ratio_bone_anti: float
    target_to_clutter_bone: float
    target_to_clutter_void: float
    onset_sample: int
    passed: bool


def scene_fidelity(
    scene_id: str,
    bscans: dict[str, np.ndarray],
    row: dict,
    dt_s: float,
    gate: dict | None = None,
) -> SceneFidelity:
    """Compute the paired-polarity metrics for one scene. ``bscans`` needs all four types."""
    g = {**DEFAULT_GATE, **(gate or {})}
    null = bscans["null"]
    resp = {t: target_response(bscans[t], null) for t in ("bone", "void", "anti_bone")}

    col = int(round(apex_trace_index(row)))
    period = int(round(1.0 / (float(row["freq_mhz"]) * 1e6) / dt_s))
    traces = {t: r[:, col] for t, r in resp.items()}
    onset = min(onset_sample(tr, g["onset_fraction"]) for tr in traces.values())
    w = top_window(onset, period, null.shape[0])

    c_ba = ncc(traces["bone"][w], traces["anti_bone"][w])
    c_va = ncc(traces["void"][w], traces["anti_bone"][w])
    amp = float(np.abs(traces["bone"][w]).max() / max(np.abs(traces["anti_bone"][w]).max(), 1e-30))

    clutter = float(np.percentile(np.abs(remove_background(null, trace_axis=1)), 99))
    clutter = max(clutter, 1e-30)
    tcr_b = float(np.abs(resp["bone"]).max() / clutter)
    tcr_v = float(np.abs(resp["void"]).max() / clutter)

    lo, hi = g["amp_ratio_range"]
    passed = c_ba <= g["max_ncc_bone_anti"] and c_va >= g["min_ncc_void_anti"] and lo <= amp <= hi
    return SceneFidelity(scene_id, c_ba, c_va, amp, tcr_b, tcr_v, onset, passed)


def run_gate(manifest_path: Path, raw_dir: Path, gate: dict | None = None) -> list[SceneFidelity]:
    """Evaluate every complete scene in the manifest. Incomplete scenes are skipped."""
    results = []
    for scene_id, twins in sorted(
        group_by_scene(list(csv.DictReader(manifest_path.open()))).items()
    ):
        paths = {t: merged_path(raw_dir, r["in_file"]) for t, r in twins.items()}
        if set(paths) != {"void", "bone", "anti_bone", "null"} or not all(
            p.exists() for p in paths.values()
        ):
            continue
        loaded = {t: read_merged_bscan(p) for t, p in paths.items()}
        dt = loaded["null"][1]
        bscans = {t: b for t, (b, _) in loaded.items()}
        results.append(scene_fidelity(scene_id, bscans, twins["bone"], dt, gate))
    return results


def write_report(results: list[SceneFidelity], path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)


def main() -> None:
    from src.data.generate_gprmax_models import default_out_dir, load_config

    p = argparse.ArgumentParser(description="Paired-polarity fidelity gate for synthetic scenes.")
    p.add_argument("--config", type=Path, default=Path("configs/synthetic_controlled.yaml"))
    p.add_argument("--raw-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    raw_dir = args.raw_dir or default_out_dir(cfg)
    results = run_gate(raw_dir / "manifest.csv", raw_dir, cfg.get("fidelity_gate"))
    if not results:
        sys.exit("[fidelity] no complete scenes found")

    write_report(results, raw_dir / "fidelity_report.csv")
    for r in results:
        print(
            f"{r.scene_id:34s} ncc(bone,anti)={r.ncc_bone_anti:+.2f} "
            f"ncc(void,anti)={r.ncc_void_anti:+.2f} amp={r.amp_ratio_bone_anti:.2f} "
            f"TCR bone={r.target_to_clutter_bone:.1f} void={r.target_to_clutter_void:.1f} "
            f"{'ok' if r.passed else 'FAIL'}"
        )
    n_fail = sum(not r.passed for r in results)
    verdict = "PASS" if n_fail == 0 else "FAIL"
    print(f"FIDELITY GATE: {verdict} ({len(results) - n_fail}/{len(results)} scenes)")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
