"""Generate controlled gprMax input files for the H1 transfer experiment.

What this does
--------------
Writes gprMax ``.in`` files. The FDTD simulation itself runs separately on the CUDA box, so
this script only emits text input plus a manifest. For every scene cell in the config it
writes the same 2D scene four times, once per target type, sharing a ``scene_id`` so they
pair exactly:

    void      eps = 1          air-filled, low eps, positive top polarity (matches real cavities)
    bone      eps = bone_eps   fossil-like, high eps, negative top polarity
    anti_bone eps computed      low eps, |reflection| matched to bone, positive polarity (control)
    null      eps = host        no contrast, negative control, should be ~undetectable

The point (see experiment_03_transfer.md C0-C2 and notes/09_research_hypothesis.md, both under
docs/) is to hold geometry, depth, host, and clutter constant and vary only the target
dielectric. That is what makes H1b (polarity is the cause) a causal claim rather than a
correlation. anti_bone is the key control: same |reflection coefficient| as bone, opposite sign.

Run
---
    python -m src.data.generate_gprmax_models --config configs/synthetic_controlled.yaml

Outputs:
    data/processed/synthetic/controlled/*.in     one gprMax model per (scene, target type)
    data/processed/synthetic/controlled/manifest.csv  one row per .in, with ground truth
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from src.config import PROCESSED_DATA_DIR
from src.data.forward_model_1d import reflection_coefficient

TARGET_TYPES = ("void", "bone", "anti_bone", "null")


def matched_low_eps_target(host_eps: float, high_eps_target: float) -> float:
    """Permittivity of the ``anti_bone`` control target.

    Returns the eps of a *low*-permittivity target (eps < host) whose normal-incidence
    reflection-coefficient magnitude equals that of *high_eps_target*, but with the opposite
    (void-like, positive) polarity. This is the H1b causal control: identical |contrast|,
    flipped sign.

    Derivation: with sh = sqrt(host), sa = sqrt(eps_anti), and m = |r| of the bone target,
        (sh - sa) / (sh + sa) = m   ->   sa = sh * (1 - m) / (1 + m)
    """
    m = abs(reflection_coefficient(host_eps, high_eps_target))
    sh = math.sqrt(host_eps)
    sa = sh * (1.0 - m) / (1.0 + m)
    return sa**2


@dataclass(frozen=True)
class Scene:
    """One controlled scene cell. The four target types share everything but eps_target."""

    scene_id: str
    target_type: str
    eps_target: float
    host_name: str
    host_eps: float
    depth_m: float
    radius_m: float
    freq_mhz: float
    x_center_m: float


def _resolve_eps(target_type: str, host_eps: float, bone_eps: float, void_eps: float) -> float:
    if target_type == "void":
        return void_eps
    if target_type == "bone":
        return bone_eps
    if target_type == "anti_bone":
        return matched_low_eps_target(host_eps, bone_eps)
    if target_type == "null":
        return host_eps
    raise ValueError(f"Unknown target type: {target_type}")


def build_scenes(config: dict) -> list[Scene]:
    """Expand the config grid into paired Scene objects (4 target types per cell)."""
    g = config["grid"]
    t = config["target"]
    scenes: list[Scene] = []
    for host_name, host_eps in g["hosts"].items():
        for depth in g["depths_m"]:
            for freq in g["freqs_mhz"]:
                cell = f"{host_name}_d{int(depth * 100):03d}_f{int(freq):04d}"
                for target_type in TARGET_TYPES:
                    eps = _resolve_eps(target_type, host_eps, t["bone_eps"], t["void_eps"])
                    scenes.append(
                        Scene(
                            scene_id=cell,
                            target_type=target_type,
                            eps_target=round(eps, 4),
                            host_name=host_name,
                            host_eps=float(host_eps),
                            depth_m=float(depth),
                            radius_m=float(t["radius_m"]),
                            freq_mhz=float(freq),
                            x_center_m=float(g["x_center_m"]),
                        )
                    )
    return scenes


@dataclass(frozen=True)
class Geometry:
    """Derived physical layout of a model, shared by the .in writer and the manifest."""

    domain_x: float
    domain_y: float
    dx: float
    surface_y: float
    antenna_y: float
    src_x: float
    rx_x: float
    trace_step: float
    n_traces: int
    target_x: float
    target_y: float


def _geometry(scene: Scene, config: dict) -> Geometry:
    d, a = config["domain"], config["antenna"]
    dx = float(d["dx_m"])
    domain_x = float(d["width_m"])
    domain_y = float(d["ground_depth_m"]) + float(d["air_gap_m"])
    surface_y = float(d["ground_depth_m"])  # ground occupies y in [0, surface_y]
    antenna_y = surface_y  # antenna sits on the surface

    sep = float(a["separation_m"])
    step = float(a["trace_step_m"])
    src_x0 = sep  # start the tx one separation in from the edge
    rx_x0 = src_x0 + sep
    # March the pair across the line until the rx would leave the domain (keep one sep margin).
    max_rx = domain_x - sep
    n_traces = int(math.floor((max_rx - rx_x0) / step)) + 1

    return Geometry(
        domain_x=domain_x,
        domain_y=domain_y,
        dx=dx,
        surface_y=surface_y,
        antenna_y=antenna_y,
        src_x=src_x0,
        rx_x=rx_x0,
        trace_step=step,
        n_traces=n_traces,
        target_x=scene.x_center_m,
        target_y=surface_y - scene.depth_m,  # depth measured down from the surface
    )


def build_input_file(scene: Scene, geo: Geometry, config: dict) -> str:
    """Render the gprMax v3 ``.in`` text for one model (2D, one cell thick in z).

    This follows the gprMax B-scan convention. #src_steps and #rx_steps march the antenna
    pair across the line, and you run the file with ``-n <n_traces>`` to assemble a B-scan.
    """
    t = config["target"]
    dx = geo.dx
    time_window_s = float(config["antenna"]["time_window_ns"]) * 1e-9
    freq_hz = scene.freq_mhz * 1e6
    host_sigma = float(t["host_sigma_s_m"])
    tgt_sigma = float(t["target_sigma_s_m"])

    lines = [
        f"#title: {scene.scene_id}__{scene.target_type}",
        f"#domain: {geo.domain_x:.3f} {geo.domain_y:.3f} {dx:.3f}",
        f"#dx_dy_dz: {dx:.3f} {dx:.3f} {dx:.3f}",
        f"#time_window: {time_window_s:.3e}",
        "",
        f"#material: {scene.host_eps:.4g} {host_sigma:.4g} 1 0 host",
        f"#material: {scene.eps_target:.4g} {tgt_sigma:.4g} 1 0 target",
        "",
        f"#waveform: ricker 1 {freq_hz:.4g} my_ricker",
        f"#hertzian_dipole: z {geo.src_x:.3f} {geo.antenna_y:.3f} 0 my_ricker",
        f"#rx: {geo.rx_x:.3f} {geo.antenna_y:.3f} 0",
        f"#src_steps: {geo.trace_step:.3f} 0 0",
        f"#rx_steps: {geo.trace_step:.3f} 0 0",
        "",
        f"#box: 0 0 0 {geo.domain_x:.3f} {geo.surface_y:.3f} {dx:.3f} host",
        "#cylinder: "
        f"{geo.target_x:.3f} {geo.target_y:.3f} 0 {geo.target_x:.3f} {geo.target_y:.3f} {dx:.3f} "
        f"{scene.radius_m:.3f} target",
        "",
        f"#geometry_view: 0 0 0 {geo.domain_x:.3f} {geo.domain_y:.3f} {dx:.3f} "
        f"{dx:.3f} {dx:.3f} {dx:.3f} {scene.scene_id}__{scene.target_type} n",
    ]
    return "\n".join(lines) + "\n"


def generate(config: dict, out_dir: Path) -> Path:
    """Write all .in files + the manifest. Returns the manifest path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = build_scenes(config)

    manifest_rows: list[dict] = []
    for scene in scenes:
        geo = _geometry(scene, config)
        in_name = f"{scene.scene_id}__{scene.target_type}.in"
        (out_dir / in_name).write_text(build_input_file(scene, geo, config))
        row = asdict(scene)
        row.update(
            {
                "in_file": in_name,
                "n_traces": geo.n_traces,
                "trace_step_m": geo.trace_step,
                "domain_x_m": geo.domain_x,
                "domain_y_m": geo.domain_y,
                "surface_y_m": geo.surface_y,
                "target_x_m": geo.target_x,
                "target_y_m": geo.target_y,
                "src_x0_m": geo.src_x,
                "rx_x0_m": geo.rx_x,
            }
        )
        manifest_rows.append(row)

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate controlled gprMax input files for H1.")
    p.add_argument("--config", type=Path, default=Path("configs/synthetic_controlled.yaml"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "synthetic" / "controlled",
        help="Where to write the .in files and manifest.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    manifest = generate(config, args.out_dir)
    n = sum(1 for _ in manifest.open()) - 1
    print(f"[generate_gprmax_models] wrote {n} models + manifest -> {manifest}")
    print("Next: run these with gprMax on the CUDA box, then process B-scans (see experiment_03).")


if __name__ == "__main__":
    main()
