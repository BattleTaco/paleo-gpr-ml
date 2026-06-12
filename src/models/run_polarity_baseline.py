"""Run the polarity matched-filter baseline and save a figure plus a results table.

This is the experiment for Cbase in experiment_03. It answers two things on controlled
forward-model traces:

    1. Does the matched filter detect each target type and read its polarity correctly?
       Expected: void +, bone -, anti-bone + (matched magnitude to bone, flipped sign), null none.
    2. How far does it hold up as noise grows? (a detection-rate and polarity-accuracy sweep)

If the polarity split is clean here, the baseline gives me a non-ML reference to compare the
learned detectors against, and direct evidence for H1b (polarity is a real, readable signal).

Run:
    python -m src.models.run_polarity_baseline
"""

from __future__ import annotations

import csv

import numpy as np

from src.config import FIGURES_DIR, RESULTS_DIR
from src.data.forward_model_1d import Layer, synthesize_trace
from src.data.generate_gprmax_models import matched_low_eps_target
from src.models.polarity_matched_filter import detect_trace

EPS_HOST = 4.0  # dry sand
EPS_BONE = 9.0
EPS_VOID = 1.0
FREQ_MHZ = 400.0
DEPTH_M = 0.5

# target -> (eps, expected polarity). None means "should not be detected".
TARGETS = {
    "void": (EPS_VOID, 1),
    "bone": (EPS_BONE, -1),
    "anti_bone": (matched_low_eps_target(EPS_HOST, EPS_BONE), 1),
    "null": (EPS_HOST, 0),
}


def _trace(target_eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Single top-interface trace (host over a half-space target)."""
    scene = [Layer("host", EPS_HOST, 0.0), Layer("target", target_eps, DEPTH_M)]
    time_ns, trace, _ = synthesize_trace(scene, center_freq_mhz=FREQ_MHZ)
    return time_ns, trace


def _noise_sweep(
    target_eps: float,
    expected_pol: int,
    rng: np.random.Generator,
    fractions: np.ndarray,
    n_trials: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Detection rate and polarity accuracy vs added Gaussian noise.

    Noise std is set as a fraction of the clean trace's peak amplitude.
    """
    _, clean = _trace(target_eps)
    peak = np.max(np.abs(clean))
    det_rate = np.zeros(fractions.size)
    pol_acc = np.zeros(fractions.size)
    for i, frac in enumerate(fractions):
        det, pol_ok = 0, 0
        for _ in range(n_trials):
            noisy = clean + rng.normal(0.0, frac * peak, size=clean.size)
            d = detect_trace(noisy, center_freq_mhz=FREQ_MHZ)
            if d.detected:
                det += 1
                if d.polarity == expected_pol:
                    pol_ok += 1
        det_rate[i] = det / n_trials
        pol_acc[i] = (pol_ok / det) if det else 0.0
    return det_rate, pol_acc


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)

    # Clean detection table
    rows = []
    print("\n=== Polarity matched filter on clean traces (dry sand, 400 MHz) ===")
    print(f"{'target':12} {'eps':>7} {'detected':>9} {'polarity':>9} {'score':>7} {'expected':>9}")
    for name, (eps, expected) in TARGETS.items():
        _, trace = _trace(eps)
        d = detect_trace(trace, center_freq_mhz=FREQ_MHZ)
        ok = d.polarity == expected
        print(
            f"{name:12} {eps:7.3f} {str(d.detected):>9} {d.polarity:>9} {d.score:>7.3f} "
            f"{expected:>9}  {'OK' if ok else 'MISMATCH'}"
        )
        rows.append(
            {
                "target": name,
                "eps": round(eps, 4),
                "detected": d.detected,
                "polarity": d.polarity,
                "expected_polarity": expected,
                "ncc_score": round(d.score, 4),
                "correct": ok,
            }
        )

    RESULTS_DIR.joinpath("tables").mkdir(parents=True, exist_ok=True)
    table_path = RESULTS_DIR / "tables" / "polarity_baseline.csv"
    with table_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved table -> {table_path}")

    # Figure: traces + noise robustness
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    for name, (eps, _) in TARGETS.items():
        time_ns, trace = _trace(eps)
        ax.plot(trace, time_ns, label=f"{name} (eps={eps:.2g})", lw=1.7)
    ax.invert_yaxis()
    ax.set_xlabel("amplitude")
    ax.set_ylabel("two-way time (ns)")
    ax.set_title("Top-interface traces, dry sand, 400 MHz\nbone reflects negative, void positive")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    fractions = np.linspace(0.0, 1.2, 13)
    for name in ("bone", "void"):
        eps, expected = TARGETS[name]
        det_rate, pol_acc = _noise_sweep(eps, expected, rng, fractions)
        ax.plot(fractions, det_rate, marker="o", label=f"{name}: detection rate")
        ax.plot(fractions, pol_acc, marker="s", ls="--", label=f"{name}: polarity accuracy")
    ax.set_xlabel("noise std / signal peak")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Robustness to noise\n(300 trials per point)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Physics baseline: polarity matched filter (Cbase)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "polarity_baseline.png"
    fig.savefig(out, dpi=150)
    print(f"Saved figure -> {out}")

    # Verdicts
    print("\n=== Verdicts ===")
    all_ok = all(r["correct"] for r in rows)
    for r in rows:
        print(
            f"[{'PASS' if r['correct'] else 'FAIL'}] {r['target']}: "
            f"polarity {r['polarity']} vs expected {r['expected_polarity']}"
        )
    print(f"[{'PASS' if all_ok else 'FAIL'}] all target types classified correctly on clean traces")


if __name__ == "__main__":
    main()
