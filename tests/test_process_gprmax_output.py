"""Tests for the gprMax output converter: rendering variants and measured-response labels.

HDF5 reads run against real gprMax output and are exercised by the pipeline, not here. The label
math is where a silent error costs the most (the pilot's analytic boxes missed every 400 MHz
target), so it is tested on synthetic responses with a known location.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.preprocess_gpr import remove_background
from src.data.process_gprmax_output import (
    apex_trace_index,
    bbox_from_response,
    load_manifest,
    render,
    target_response,
    time_gain,
    to_uint8_image,
)

ROW = {
    "in_file": "s__bone.in",
    "target_type": "bone",
    "depth_m": "0.1",
    "trace_step_m": "0.004",
    "target_x_m": "0.25",
    "src_x0_m": "0.04",
    "rx_x0_m": "0.08",
}
N_SAMPLES, N_TRACES = 1000, 96


def _row(**over):
    return {**ROW, **over}


def _blob(t0: int, t1: int, c0: int, c1: int) -> np.ndarray:
    r = np.zeros((N_SAMPLES, N_TRACES))
    r[t0:t1, c0:c1] = 1.0
    return r


def test_time_gain_boosts_deeper_samples():
    g = time_gain(np.ones((100, 10)), power=2.0)
    assert g[-1].mean() > g[0].mean() and g[0].mean() == 0.0


def test_to_uint8_image_range_dtype_and_zero():
    img = to_uint8_image(np.random.default_rng(0).normal(size=(200, 50)))
    assert img.dtype == np.uint8 and img.shape == (200, 50)
    assert np.all(to_uint8_image(np.zeros((10, 10))) == 0)


def test_remove_background_kills_flat_events_keeps_local_ones():
    b = np.zeros((50, 20))
    b[10] = 5.0  # direct wave: identical on every trace
    b[30, 8] = 1.0  # a local target
    out = remove_background(b, trace_axis=1)
    assert np.allclose(out[10], 0.0)
    assert out[30, 8] > 0.9
    with pytest.raises(ValueError):
        remove_background(b, method="nope")


def test_bgr_variant_suppresses_direct_wave_relative_to_raw():
    # A direct wave spanning many samples (like a real wavelet) over a small noise floor.
    b = np.random.default_rng(0).normal(scale=0.01, size=(400, 40))
    b[150:250] += 10.0
    b[300, 20] = 1.0
    raw = render(b, {"background_removal": None}).astype(float)
    bgr = render(b, {"background_removal": "mean"}).astype(float)
    # In raw, the late target is weak relative to the clip level; bgr makes it saturate.
    assert abs(bgr[300, 20] - 127.5) > abs(raw[300, 20] - 127.5)


def test_apex_trace_index_centered():
    # midpoint starts at 0.06 m, target at 0.25 m, 4 mm step -> 47.5
    assert apex_trace_index(ROW) == pytest.approx(47.5)


def test_bbox_tracks_measured_response_in_time():
    # Response at samples 600-700 near the apex: the box must cover exactly that time span,
    # wherever travel-time formulas would have put it.
    _, cx, cy, w, h = bbox_from_response(_blob(600, 700, 45, 50), ROW, envelope_threshold=0.5)
    assert cy * N_SAMPLES == pytest.approx(650, abs=1)
    assert h * N_SAMPLES == pytest.approx(100, abs=1)
    assert cx * N_TRACES == pytest.approx(47.5, abs=0.5)
    # Lateral half width is one depth: 0.1 m / 4 mm = 25 traces.
    assert w * N_TRACES == pytest.approx(50, abs=1)


def test_bbox_clips_at_image_edge():
    row = _row(target_x_m="0.08")  # apex near the left edge
    _, cx, cy, w, h = bbox_from_response(_blob(100, 200, 0, 10), row)
    assert cx - w / 2 >= -1e-9 and cx + w / 2 <= 1 + 1e-9
    assert 0 < w < 0.5


def test_bbox_rejects_zero_response():
    with pytest.raises(ValueError):
        bbox_from_response(np.zeros((N_SAMPLES, N_TRACES)), ROW)


def test_target_response_requires_matching_twins():
    with pytest.raises(ValueError):
        target_response(np.zeros((10, 5)), np.zeros((10, 6)))


def test_load_manifest_keeps_null_target_type(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("stem,target_type\na,bone\nb,null\n")
    df = load_manifest(p)
    assert list(df.target_type) == ["bone", "null"]
