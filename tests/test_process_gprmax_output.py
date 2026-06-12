"""Tests for the gprMax output converter (image and box math).

The HDF5 read runs on the CUDA box against real output, so it is not tested here. The image
normalization and the bounding-box geometry are pure functions, and those are what could be
silently wrong, so I test them on synthetic arrays and a synthetic manifest row.
"""

from __future__ import annotations

import numpy as np

from src.data.process_gprmax_output import bbox_yolo, time_gain, to_uint8_image

# A row shaped like what generate_gprmax_models writes (dry_sand, 10 cm deep, 400 MHz, bone).
ROW_BONE = {
    "target_type": "bone",
    "depth_m": "0.1",
    "host_eps": "4.0",
    "trace_step_m": "0.004",
    "n_traces": "96",
    "target_x_m": "0.25",
    "src_x0_m": "0.04",
    "rx_x0_m": "0.08",
}
DT_S = 4.7e-12
N_SAMPLES = 2553


def _row(**over):
    r = dict(ROW_BONE)
    r.update(over)
    return r


def test_time_gain_boosts_deeper_samples():
    b = np.ones((100, 10))
    g = time_gain(b, power=2.0)
    assert g.shape == b.shape
    assert g[-1].mean() > g[0].mean()  # later (deeper) rows scaled up
    assert g[0].mean() == 0.0  # first sample gain is 0


def test_to_uint8_image_range_and_dtype():
    rng = np.random.default_rng(0)
    b = rng.normal(size=(200, 50))
    img = to_uint8_image(b)
    assert img.dtype == np.uint8
    assert img.shape == b.shape
    assert img.min() >= 0 and img.max() <= 255


def test_to_uint8_image_zeros():
    img = to_uint8_image(np.zeros((10, 10)))
    assert np.all(img == 0)


def test_bbox_null_has_no_box():
    assert bbox_yolo(_row(target_type="null"), N_SAMPLES, DT_S) is None


def test_bbox_bone_is_valid_and_centered():
    box = bbox_yolo(ROW_BONE, N_SAMPLES, DT_S)
    assert box is not None
    cls, cx, cy, w, h = box
    assert cls == 0
    for v in (cx, cy, w, h):
        assert 0.0 <= v <= 1.0
    # Target sits at x=0.25 in a 0.04..~0.46 survey, so the apex is near the middle.
    assert 0.4 < cx < 0.6
    # 10 cm deep target gives a shallow apex, so the box center is in the upper half.
    assert cy < 0.5
    assert w > 0 and h > 0


def test_bbox_deeper_target_lower_in_image():
    shallow = bbox_yolo(_row(depth_m="0.1"), N_SAMPLES, DT_S)
    deep = bbox_yolo(_row(depth_m="0.25"), N_SAMPLES, DT_S)
    assert deep[2] > shallow[2]  # deeper target -> larger cy (further down the image)
