"""Generate controlled gprMax input files for the H1 transfer experiment.

What this does
--------------
Writes gprMax ``.in`` files. The FDTD simulation runs separately (``src.data.run_gprmax_batch``),
so this script only emits text input plus a manifest. For every scene cell in the config it
writes the same 2D scene four times, once per target type, sharing a ``scene_id`` so they
pair exactly:

    void      eps = 1          air-filled, low eps, positive top polarity (matches real cavities)
    bone      eps = bone_eps   fossil-like, high eps, negative top polarity
    anti_bone eps computed      low eps, |reflection| matched to bone, positive polarity (control)
    null      eps = host        no contrast, negative control, should be ~undetectable

The point (see docs/experiments/experiment_03_transfer.md C0-C2) is to hold geometry, depth,
host, and clutter constant and vary only the target dielectric. anti_bone is the key control:
same |reflection coefficient| as bone, opposite sign.

A scene cell is (host, depth, freq, x_center, clutter realization). Clutter is a set of small
scatterers drawn from an RNG seeded by the cell key, which excludes the target type, so all four
target types in a cell get the byte-identical clutter layout. The null twin is then an exact
reference: ``bscan(target) - bscan(null)`` is the target's own response, which downstream code
uses for labels and the fidelity gate.

Run
---
    python -m src.data.generate_gprmax_models --config configs/synthetic_controlled.yaml

Outputs (default dir: data/processed/synthetic/<experiment.name>/):
    *.in           one gprMax model per (scene, target type)
    manifest.csv   one row per .in, with ground truth
    clutter.json   the scatterer list per scene_id, for provenance
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from src.config import PROCESSED_DATA_DIR
from src.data.forward_model_1d import reflection_coefficient

TARGET_TYPES = ("void", "bone", "anti_bone", "null")

# gprMax's default PML is 10 cells on each face. Keep scatterers out of it.
PML_CELLS = 10


def matched_low_eps_target(host_eps: float, high_eps_target: float) -> float:
    """Permittivity of the ``anti_bone`` control target.

    Returns the eps of a *low*-permittivity target (eps < host) whose normal-incidence
    reflection-coefficient magnitude equals that of *high_eps_target*, but with the opposite
    (void-like, positive) polarity. This is the H1b causal control: identical |contrast|,
    flipped sign.

    Derivation: with sh = sqrt(host), sa = sqrt(eps_anti), and m = |r| of the bone target,
        (sh - sa) / (sh + sa) = m   ->   sa = sh * (1 - m) / (1 + m)
    """
    if high_eps_target <= host_eps:
        raise ValueError(
            f"anti_bone needs bone eps > host eps (got bone {high_eps_target}, host {host_eps}). "
            "In a wetter host the bone polarity itself flips and the control is undefined."
        )
    m = abs(reflection_coefficient(host_eps, high_eps_target))
    sh = math.sqrt(host_eps)
    sa = sh * (1.0 - m) / (1.0 + m)
    return sa**2


@dataclass(frozen=True)
class Scene:
    """One model. The four target types of a scene_id share everything but eps_target."""

    scene_id: str
    target_type: str
    eps_target: float
    host_name: str
    host_eps: float
    depth_m: float
    radius_m: float
    freq_mhz: float
    x_center_m: float
    clutter_realization: int


@dataclass(frozen=True)
class Scatterer:
    """A small clutter cylinder (pebble, nodule, root cross-section)."""

    x_m: float
    y_m: float
    radius_m: float
    eps: float


def _x_centers(grid: dict) -> list[float]:
    # Accept the old scalar key so pilot configs still load.
    if "x_centers_m" in grid:
        return [float(x) for x in grid["x_centers_m"]]
    return [float(grid["x_center_m"])]


def scene_cell_id(
    host_name: str, depth_m: float, freq_mhz: float, x_m: float, realization: int
) -> str:
    """Stable, sortable id for a scene cell. Shared by all four target types."""
    return (
        f"{host_name}_d{round(depth_m * 100):03d}_f{round(freq_mhz):04d}"
        f"_x{round(x_m * 1000):04d}_c{realization:02d}"
    )


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
    n_real = int(config.get("clutter", {}).get("realizations", 0))
    # Realization 0 is clutter-free only when clutter is off; otherwise every realization has
    # clutter. This keeps a clean pilot reproducible with the same code path.
    realizations = range(max(n_real, 1))

    scenes: list[Scene] = []
    for host_name, host_eps in g["hosts"].items():
        for depth in g["depths_m"]:
            for freq in g["freqs_mhz"]:
                for x in _x_centers(g):
                    for k in realizations:
                        cell = scene_cell_id(host_name, float(depth), float(freq), x, k)
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
                                    x_center_m=x,
                                    clutter_realization=k,
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
    n_traces = int(math.floor((max_rx - rx_x0) / step + 1e-9)) + 1

    target_y = surface_y - scene.depth_m  # depth of the target center below the surface
    if target_y - scene.radius_m < PML_CELLS * dx:
        raise ValueError(f"{scene.scene_id}: target reaches into the bottom PML; deepen domain.")
    if target_y + scene.radius_m >= surface_y:
        raise ValueError(f"{scene.scene_id}: target breaks the surface; increase depth.")

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
        target_y=target_y,
    )


def _cell_rng(scene: Scene, seed: int) -> np.random.Generator:
    """RNG keyed on the scene cell only (never the target type), so twins share clutter."""
    return np.random.default_rng([int(seed), zlib.crc32(scene.scene_id.encode())])


def build_clutter(scene: Scene, geo: Geometry, config: dict) -> list[Scatterer]:
    """Place non-overlapping clutter scatterers in the ground, clear of the target and PML.

    Rejection sampling with a fixed attempt budget. Raises instead of silently returning fewer
    scatterers, so a config that cannot be satisfied fails at generation, not at analysis.
    """
    c = config.get("clutter", {})
    if int(c.get("realizations", 0)) <= 0:
        return []

    rng = _cell_rng(scene, config["experiment"]["seed"])
    n_lo, n_hi = c["n_range"]
    r_lo, r_hi = c["radius_range_m"]
    e_lo, e_hi = c["eps_scale_range"]
    gap = float(c["min_gap_m"])
    pml = PML_CELLS * geo.dx

    n = int(rng.integers(n_lo, n_hi + 1))
    placed: list[Scatterer] = []
    attempts = 0
    while len(placed) < n:
        attempts += 1
        if attempts > 200 * n:
            raise RuntimeError(f"{scene.scene_id}: could not place {n} scatterers; relax clutter.")
        r = float(rng.uniform(r_lo, r_hi))
        x = float(rng.uniform(pml + r, geo.domain_x - pml - r))
        y = float(rng.uniform(pml + r, geo.surface_y - r - gap))
        if math.hypot(x - geo.target_x, y - geo.target_y) < scene.radius_m + r + gap:
            continue
        if any(math.hypot(x - s.x_m, y - s.y_m) < s.radius_m + r + gap for s in placed):
            continue
        eps = max(1.0, scene.host_eps * float(rng.uniform(e_lo, e_hi)))
        placed.append(Scatterer(round(x, 4), round(y, 4), round(r, 4), round(eps, 3)))
    return placed


def build_input_file(
    scene: Scene, geo: Geometry, config: dict, clutter: Sequence[Scatterer] = ()
) -> str:
    """Render the gprMax v3 ``.in`` text for one model (2D, one cell thick in z).

    Follows the gprMax B-scan convention: #src_steps / #rx_steps march the antenna pair, and the
    file is run with ``-n <n_traces>``. Objects later in the file overwrite earlier ones, so the
    order is host, clutter, target.
    """
    t = config["target"]
    dx = geo.dx
    time_window_s = float(config["antenna"]["time_window_ns"]) * 1e-9
    freq_hz = scene.freq_mhz * 1e6
    host_sigma = float(t["host_sigma_s_m"])
    tgt_sigma = float(t["target_sigma_s_m"])
    name = f"{scene.scene_id}__{scene.target_type}"

    lines = [
        f"#title: {name}",
        f"#domain: {geo.domain_x:.3f} {geo.domain_y:.3f} {dx:.3f}",
        f"#dx_dy_dz: {dx:.3f} {dx:.3f} {dx:.3f}",
        f"#time_window: {time_window_s:.3e}",
        "",
        f"#material: {scene.host_eps:.4g} {host_sigma:.4g} 1 0 host",
        f"#material: {scene.eps_target:.4g} {tgt_sigma:.4g} 1 0 target",
    ]
    lines += [
        f"#material: {s.eps:.4g} {host_sigma:.4g} 1 0 clutter{i}" for i, s in enumerate(clutter)
    ]
    lines += [
        "",
        f"#waveform: ricker 1 {freq_hz:.4g} my_ricker",
        f"#hertzian_dipole: z {geo.src_x:.3f} {geo.antenna_y:.3f} 0 my_ricker",
        f"#rx: {geo.rx_x:.3f} {geo.antenna_y:.3f} 0",
        f"#src_steps: {geo.trace_step:.3f} 0 0",
        f"#rx_steps: {geo.trace_step:.3f} 0 0",
        "",
        f"#box: 0 0 0 {geo.domain_x:.3f} {geo.surface_y:.3f} {dx:.3f} host",
    ]
    lines += [
        f"#cylinder: {s.x_m:.4f} {s.y_m:.4f} 0 {s.x_m:.4f} {s.y_m:.4f} {dx:.3f} "
        f"{s.radius_m:.4f} clutter{i}"
        for i, s in enumerate(clutter)
    ]
    lines.append(
        "#cylinder: "
        f"{geo.target_x:.3f} {geo.target_y:.3f} 0 {geo.target_x:.3f} {geo.target_y:.3f} {dx:.3f} "
        f"{scene.radius_m:.3f} target"
    )
    # gprMax writes the geometry view once per trace with -n, which dominated disk use in the
    # pilot (~80%). Off by default; turn on to debug a scene layout.
    if config.get("output", {}).get("geometry_view", False):
        lines += [
            "",
            f"#geometry_view: 0 0 0 {geo.domain_x:.3f} {geo.domain_y:.3f} {dx:.3f} "
            f"{dx:.3f} {dx:.3f} {dx:.3f} {name} n",
        ]
    return "\n".join(lines) + "\n"


def default_out_dir(config: dict) -> Path:
    return PROCESSED_DATA_DIR / "synthetic" / config["experiment"]["name"]


def generate(config: dict, out_dir: Path) -> Path:
    """Write all .in files, the manifest, and clutter.json. Returns the manifest path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = build_scenes(config)

    manifest_rows: list[dict] = []
    clutter_by_cell: dict[str, list[dict]] = {}
    for scene in scenes:
        geo = _geometry(scene, config)
        clutter = build_clutter(scene, geo, config)
        clutter_by_cell.setdefault(scene.scene_id, [asdict(s) for s in clutter])

        in_name = f"{scene.scene_id}__{scene.target_type}.in"
        (out_dir / in_name).write_text(build_input_file(scene, geo, config, clutter))
        row = asdict(scene)
        row.update(
            {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in {
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
                    "n_clutter": len(clutter),
                }.items()
            }
        )
        manifest_rows.append(row)

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    (out_dir / "clutter.json").write_text(json.dumps(clutter_by_cell, indent=1))
    return manifest_path


def load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate controlled gprMax input files for H1.")
    p.add_argument("--config", type=Path, default=Path("configs/synthetic_controlled.yaml"))
    p.add_argument("--out-dir", type=Path, default=None, help="Default: synthetic/<exp name>")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out_dir = args.out_dir or default_out_dir(config)
    manifest = generate(config, out_dir)
    n = sum(1 for _ in manifest.open()) - 1
    print(f"[generate_gprmax_models] wrote {n} models + manifest -> {manifest}")


if __name__ == "__main__":
    main()
