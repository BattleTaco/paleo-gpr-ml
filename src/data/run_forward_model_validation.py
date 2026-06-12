"""Run the 1D forward-model physics validation and save the figure + summary.

This is the experiment, not the library. It uses src/data/forward_model_1d.py to answer:

    1. Does a buried bone produce the kind of reflection signature Peredo et al. describe?
    2. Are bone (high permittivity) and an air cavity (low permittivity) polarity-inverted?
    3. How does the signature change with depth, host medium, and antenna frequency?

Output:
    results/figures/forward_model_1d_polarity.png   (the validation figure)
    a printed verification table (also the source of the numbers in the write-up)

Run:
    python -m src.data.run_forward_model_validation
"""

from __future__ import annotations

import numpy as np

from src.config import FIGURES_DIR
from src.data.forward_model_1d import (
    buried_target_scene,
    reflection_coefficient,
    synthesize_trace,
)

# Dielectric values from the reading notes (Peredo 2024, Catanzariti 2023).
EPS = {
    "dry sand": 4.0,
    "limestone": 6.0,
    "bone": 9.0,  # mineralized bone, mid of the 7-12 range
    "air (void)": 1.0,
    "rock (no contrast)": 4.3,  # nearly identical to sand, so a near-invisible target
}


def _peak_polarity(time_ns: np.ndarray, trace: np.ndarray, twt: float, win_ns: float = 1.5):
    """Sign and value of the largest-|amplitude| sample near an interface's TWT."""
    mask = np.abs(time_ns - twt) <= win_ns
    seg = trace[mask]
    if seg.size == 0:
        return 0.0, 0.0
    k = int(np.argmax(np.abs(seg)))
    val = float(seg[k])
    return np.sign(val), val


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    host = EPS["dry sand"]
    # 40 cm slab. Thick enough that the top and bottom reflections resolve at 400 MHz
    # (wavelength ~0.37 m in dry sand), so the negative-top, positive-bottom structure is
    # actually visible. Thinner targets fall below tuning thickness and the two reflections
    # merge into one composite wiggle, which I note in the write-up.
    target_top, target_thick = 0.6, 0.40

    # Scenes: target type -> layered model, all in dry sand at the same depth.
    scenes = {
        "Bone (eps=9)": buried_target_scene(host, EPS["bone"], target_top, target_thick),
        "Air cavity (eps=1)": buried_target_scene(
            host, EPS["air (void)"], target_top, target_thick
        ),
        "Rock (eps=4.3, ~no contrast)": buried_target_scene(
            host, EPS["rock (no contrast)"], target_top, target_thick
        ),
    }

    print("\n=== Verification: interface reflection coefficients (dry sand host) ===")
    print(f"{'target':30} {'r_top (sand->tgt)':>18} {'r_bottom (tgt->sand)':>22}")
    for name, eps in [
        ("bone", EPS["bone"]),
        ("air void", EPS["air (void)"]),
        ("rock", EPS["rock (no contrast)"]),
    ]:
        rt = reflection_coefficient(host, eps)
        rb = reflection_coefficient(eps, host)
        print(f"{name:30} {rt:18.3f} {rb:22.3f}")

    # Figure: 3 panels.
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Panel 1: the three target types over the same column, traces overlaid.
    ax = axes[0]
    print("\n=== Verification: peak polarity at each interface (dry sand host) ===")
    for name, scene in scenes.items():
        time_ns, trace, interfaces = synthesize_trace(scene, center_freq_mhz=400.0)
        ax.plot(trace, time_ns, label=name, lw=1.8)
        top_twt, bot_twt = interfaces[0][0], interfaces[1][0]
        s_top, v_top = _peak_polarity(time_ns, trace, top_twt)
        s_bot, v_bot = _peak_polarity(time_ns, trace, bot_twt)
        pt = "+" if s_top >= 0 else "-"
        pb = "+" if s_bot >= 0 else "-"
        print(f"{name:32} top: {pt} ({v_top:+.3f})   bottom: {pb} ({v_bot:+.3f})")
    ax.axhline(2 * target_top / (0.299792458 / np.sqrt(host)), color="gray", ls="--", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("amplitude")
    ax.set_ylabel("two-way time (ns)")
    ax.set_title("Bone vs cavity vs rock\n(same depth, dry sand, 400 MHz)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # Panel 2: bone vs cavity, zoomed in on the polarity inversion, side by side as wiggles.
    ax = axes[1]
    t_b, tr_b, _ = synthesize_trace(scenes["Bone (eps=9)"], center_freq_mhz=400.0)
    t_c, tr_c, _ = synthesize_trace(scenes["Air cavity (eps=1)"], center_freq_mhz=400.0)
    ax.plot(tr_b, t_b, label="bone", color="C0", lw=2)
    ax.plot(tr_c, t_c, label="cavity", color="C1", lw=2)
    ax.fill_betweenx(t_b, 0, tr_b, where=(tr_b > 0), color="C0", alpha=0.2)
    ax.fill_betweenx(t_c, 0, tr_c, where=(tr_c > 0), color="C1", alpha=0.2)
    ax.invert_yaxis()
    ax.set_xlabel("amplitude")
    ax.set_ylabel("two-way time (ns)")
    ax.set_title("Polarity inversion:\nbone (high eps) vs cavity (low eps)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: bone at three depths (signature shifts later, stays same shape).
    ax = axes[2]
    for depth in (0.3, 0.6, 1.0):
        scene = buried_target_scene(host, EPS["bone"], depth, target_thick)
        time_ns, trace, _ = synthesize_trace(scene, center_freq_mhz=400.0)
        ax.plot(trace, time_ns, label=f"bone @ {depth:.1f} m", lw=1.6)
    ax.invert_yaxis()
    ax.set_xlabel("amplitude")
    ax.set_ylabel("two-way time (ns)")
    ax.set_title("Bone signature vs depth\n(dry sand, 400 MHz)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "1D GPR forward-model validation: reflection polarity of fossil-like targets",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "forward_model_1d_polarity.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure -> {out}")

    # Automated verdicts.
    r_bone_top = reflection_coefficient(host, EPS["bone"])
    r_cav_top = reflection_coefficient(host, EPS["air (void)"])
    v_bone = "PASS" if r_bone_top < 0 else "FAIL"
    v_cav = "PASS" if r_cav_top > 0 else "FAIL"
    v_inv = "PASS" if np.sign(r_bone_top) == -np.sign(r_cav_top) else "FAIL"
    print("\n=== Verdicts ===")
    print(f"[{v_bone}] bone top reflects negative (high-eps target)")
    print(f"[{v_cav}] cavity top reflects positive (low-eps target)")
    print(f"[{v_inv}] bone and cavity are polarity-inverted")
    print(
        f"     |r_bone_top|={abs(r_bone_top):.3f}  |r_cavity_top|={abs(r_cav_top):.3f}  "
        f"(cavity contrast is {abs(r_cav_top) / abs(r_bone_top):.1f}x stronger)"
    )


if __name__ == "__main__":
    main()
