"""Tests for the controlled gprMax generator.

The critical invariant for the experiment is that the four target types in a scene differ
in nothing but the target dielectric, which is what makes H1b causal. I test that here,
plus the anti-bone polarity-matching physics and the structural validity of the .in files.
"""

from __future__ import annotations

import pytest

from src.data.forward_model_1d import reflection_coefficient
from src.data.generate_gprmax_models import (
    TARGET_TYPES,
    Scene,
    _geometry,
    build_input_file,
    build_scenes,
    generate,
    matched_low_eps_target,
)

CONFIG = {
    "experiment": {"name": "test", "seed": 42},
    "domain": {"air_gap_m": 0.04, "ground_depth_m": 0.30, "width_m": 0.50, "dx_m": 0.002},
    "antenna": {"separation_m": 0.04, "trace_step_m": 0.004, "time_window_ns": 12.0},
    "target": {
        "bone_eps": 9.0,
        "void_eps": 1.0,
        "radius_m": 0.05,
        "host_sigma_s_m": 0.005,
        "target_sigma_s_m": 0.0,
    },
    "grid": {
        "hosts": {"dry_sand": 4.0, "limestone": 6.0},
        "depths_m": [0.10, 0.20],
        "freqs_mhz": [400, 800],
        "x_center_m": 0.25,
    },
}


def test_anti_bone_matches_bone_magnitude_opposite_sign():
    host, bone = 4.0, 9.0
    eps_anti = matched_low_eps_target(host, bone)
    r_bone = reflection_coefficient(host, bone)
    r_anti = reflection_coefficient(host, eps_anti)
    # Same magnitude, opposite sign (this IS the causal control).
    assert abs(r_anti) == pytest.approx(abs(r_bone), abs=1e-6)
    assert r_bone < 0 < r_anti
    # And it must be a genuine low-eps target (below host).
    assert eps_anti < host


def test_anti_bone_specific_value_dry_sand():
    # Documented worked example: host eps 4, bone eps 9 -> anti-bone eps ~1.78.
    assert matched_low_eps_target(4.0, 9.0) == pytest.approx(1.778, abs=1e-3)


def test_null_target_equals_host():
    scenes = build_scenes(CONFIG)
    for s in scenes:
        if s.target_type == "null":
            assert s.eps_target == pytest.approx(s.host_eps)


def test_void_and_bone_eps():
    scenes = build_scenes(CONFIG)
    voids = [s for s in scenes if s.target_type == "void"]
    bones = [s for s in scenes if s.target_type == "bone"]
    assert all(s.eps_target == 1.0 for s in voids)
    assert all(s.eps_target == 9.0 for s in bones)


def test_four_target_types_per_scene_share_geometry():
    """The control invariant: within a scene_id, only eps_target varies."""
    scenes = build_scenes(CONFIG)
    by_scene: dict[str, list[Scene]] = {}
    for s in scenes:
        by_scene.setdefault(s.scene_id, []).append(s)

    for scene_id, group in by_scene.items():
        assert {s.target_type for s in group} == set(TARGET_TYPES), scene_id
        # Everything except target_type and eps_target must be identical across the group.
        for field in ("host_eps", "depth_m", "radius_m", "freq_mhz", "x_center_m"):
            values = {getattr(s, field) for s in group}
            assert len(values) == 1, f"{field} varied within scene {scene_id}: {values}"


def test_grid_size():
    # 2 hosts x 2 depths x 2 freqs x 4 target types
    assert len(build_scenes(CONFIG)) == 2 * 2 * 2 * 4


def test_geometry_target_depth_and_traces():
    scene = build_scenes(CONFIG)[0]
    geo = _geometry(scene, CONFIG)
    # Target sits 'depth' below the surface.
    assert geo.target_y == pytest.approx(geo.surface_y - scene.depth_m)
    assert geo.surface_y == pytest.approx(CONFIG["domain"]["ground_depth_m"])
    # Antenna pair stays inside the domain.
    last_rx = geo.rx_x + (geo.n_traces - 1) * geo.trace_step
    assert last_rx <= geo.domain_x
    assert geo.n_traces > 1


def test_input_file_has_required_directives():
    scene = build_scenes(CONFIG)[0]
    geo = _geometry(scene, CONFIG)
    text = build_input_file(scene, geo, CONFIG)
    for directive in (
        "#domain:",
        "#dx_dy_dz:",
        "#time_window:",
        "#material:",
        "#waveform: ricker",
        "#hertzian_dipole:",
        "#rx:",
        "#src_steps:",
        "#cylinder:",
        "#box:",
    ):
        assert directive in text, f"missing {directive}"
    # The target permittivity must actually appear in a material line.
    assert f"{scene.eps_target:.4g}" in text


def test_generate_writes_files_and_manifest(tmp_path):
    manifest = generate(CONFIG, tmp_path)
    assert manifest.exists()
    in_files = list(tmp_path.glob("*.in"))
    assert len(in_files) == len(build_scenes(CONFIG))
    # Manifest has one data row per .in file.
    rows = manifest.read_text().strip().splitlines()
    assert len(rows) - 1 == len(in_files)
