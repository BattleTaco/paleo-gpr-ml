"""Tests for the polarity matched filter (the physics baseline).

These check that the filter does what the physics says it should: detect a target, read the
correct polarity for bone vs void, treat the anti-bone control like a void (same polarity),
and reject a contrast-free trace. If these hold, the baseline is sound enough to compare ML
against.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.forward_model_1d import Layer, synthesize_trace
from src.data.generate_gprmax_models import matched_low_eps_target
from src.models.polarity_matched_filter import (
    detect_bscan,
    detect_trace,
    normalized_cross_correlation,
)

EPS_SAND = 4.0
EPS_BONE = 9.0
EPS_VOID = 1.0
FREQ = 400.0


def _trace(target_eps: float):
    # Single top interface (host over a half-space target). This isolates the top reflection,
    # which is the polarity diagnostic the matched filter is meant to read.
    scene = [Layer("host", EPS_SAND, 0.0), Layer("target", target_eps, 0.5)]
    _, trace, _ = synthesize_trace(scene, center_freq_mhz=FREQ)
    return trace


def test_ncc_is_bounded():
    trace = _trace(EPS_BONE)
    from src.data.forward_model_1d import ricker_wavelet

    ncc = normalized_cross_correlation(trace, ricker_wavelet(FREQ, 0.02, 8.0))
    assert ncc.min() >= -1.0 - 1e-9
    assert ncc.max() <= 1.0 + 1e-9


def test_bone_detected_negative_polarity():
    d = detect_trace(_trace(EPS_BONE), center_freq_mhz=FREQ)
    assert d.detected
    assert d.polarity == -1
    assert d.label == "bone-like"


def test_void_detected_positive_polarity():
    d = detect_trace(_trace(EPS_VOID), center_freq_mhz=FREQ)
    assert d.detected
    assert d.polarity == 1
    assert d.label == "void-like"


def test_anti_bone_reads_like_a_void():
    # The control: matched |contrast| to bone but low eps, so it should read positive (void-like).
    eps_anti = matched_low_eps_target(EPS_SAND, EPS_BONE)
    d = detect_trace(_trace(eps_anti), center_freq_mhz=FREQ)
    assert d.detected
    assert d.polarity == 1


def test_null_trace_not_detected():
    # Target eps == host means no contrast, so the trace is flat and nothing should fire.
    flat = _trace(EPS_SAND)
    d = detect_trace(flat, center_freq_mhz=FREQ)
    assert not d.detected
    assert d.label == "none"


def test_detect_bscan_shape_and_count():
    traces = [_trace(EPS_BONE) for _ in range(5)]
    bscan = np.stack(traces, axis=1)  # (n_samples, n_traces)
    dets = detect_bscan(bscan, center_freq_mhz=FREQ)
    assert len(dets) == 5
    assert all(d.polarity == -1 for d in dets)


def test_detect_bscan_rejects_1d():
    with pytest.raises(ValueError):
        detect_bscan(np.zeros(100))


def test_empty_template_safe():
    out = normalized_cross_correlation(np.ones(10), np.array([]))
    assert np.all(out == 0.0)
