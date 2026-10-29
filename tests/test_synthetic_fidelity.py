"""Tests for the paired-polarity fidelity gate on synthetic B-scans with a known answer."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.synthetic_fidelity import ncc, onset_sample, scene_fidelity

DT = 1e-11  # 10 ps
FREQ_MHZ = 400.0
N, M = 2000, 96
ROW = {
    "freq_mhz": str(FREQ_MHZ),
    "trace_step_m": "0.004",
    "target_x_m": "0.25",
    "src_x0_m": "0.04",
    "rx_x0_m": "0.08",
}


def _ricker(n: int, t0: int) -> np.ndarray:
    t = (np.arange(n) - t0) * DT
    a = (np.pi * FREQ_MHZ * 1e6 * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def _scene(bone_sign: float = -1.0, anti_sign: float = 1.0, void_sign: float = 1.0):
    rng = np.random.default_rng(0)
    null = rng.normal(scale=1e-3, size=(N, M))
    null[100] += 1.0  # direct wave
    pulse = _ricker(N, 900)[:, None] * np.ones((1, M))
    return {
        "null": null,
        "bone": null + bone_sign * 0.2 * pulse,
        "anti_bone": null + anti_sign * 0.2 * pulse,
        "void": null + void_sign * 0.5 * pulse,
    }


def test_ncc_basic():
    x = np.sin(np.linspace(0, 6, 100))
    assert ncc(x, x) == pytest.approx(1.0)
    assert ncc(x, -x) == pytest.approx(-1.0)
    assert ncc(x, np.zeros(100)) == 0.0


def test_onset_sample():
    tr = np.zeros(100)
    tr[40:50] = 1.0
    assert onset_sample(tr, 0.1) == 40


def test_correct_physics_passes():
    r = scene_fidelity("s", _scene(), ROW, DT)
    assert r.passed
    assert r.ncc_bone_anti == pytest.approx(-1.0, abs=1e-3)
    assert r.ncc_void_anti == pytest.approx(1.0, abs=1e-3)
    assert r.amp_ratio_bone_anti == pytest.approx(1.0, abs=1e-3)
    assert r.target_to_clutter_void > r.target_to_clutter_bone > 1


def test_no_polarity_inversion_fails():
    # If bone reflected like anti_bone, H1's premise is absent in the sim: the gate must fail.
    assert not scene_fidelity("s", _scene(bone_sign=1.0), ROW, DT).passed


def test_unmatched_amplitude_fails():
    s = _scene()
    s["anti_bone"] = s["null"] + 5 * (s["anti_bone"] - s["null"])
    assert not scene_fidelity("s", s, ROW, DT).passed
