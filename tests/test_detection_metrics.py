"""Detection metrics on hand-computable cases. These numbers decide H1a/H1b, so they are pinned."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.detection_metrics import (
    EvalSet,
    average_precision,
    bootstrap_ci,
    iou_xywh,
    match_detections,
    recall_at_fpi,
)

BOX = (0.5, 0.5, 0.2, 0.2)
FAR = (0.1, 0.1, 0.1, 0.1)


def _images():
    # Two target images (cells a, b) and one null image (cell c).
    return pd.DataFrame(
        {"image": ["a_bone", "b_bone", "c_null"], "cell": ["a", "b", "c"], "gt": [BOX, BOX, None]}
    )


def _dets(rows):
    return pd.DataFrame(rows, columns=["image", "score", "cx", "cy", "w", "h"])


def test_iou():
    assert iou_xywh(np.array([BOX]), np.array(BOX))[0] == pytest.approx(1.0)
    assert iou_xywh(np.array([FAR]), np.array(BOX))[0] == 0.0
    shifted = (0.55, 0.5, 0.2, 0.2)  # overlap 0.15 x 0.2 = 0.03, union 0.04 + 0.04 - 0.03
    assert iou_xywh(np.array([shifted]), np.array(BOX))[0] == pytest.approx(0.03 / 0.05)


def test_perfect_detector():
    es = EvalSet(_images(), _dets([("a_bone", 0.9, *BOX), ("b_bone", 0.8, *BOX)]))
    assert es.ap() == pytest.approx(1.0)
    assert es.recall_at_fpi(0.1) == pytest.approx(1.0)


def test_mislocalized_detections_are_false_positives():
    es = EvalSet(_images(), _dets([("a_bone", 0.9, *FAR), ("b_bone", 0.8, *FAR)]))
    assert es.ap() == 0.0


def test_duplicate_detection_counts_once():
    m = match_detections(_images(), _dets([("a_bone", 0.9, *BOX), ("a_bone", 0.7, *BOX)]))
    assert m["tp"].tolist() == [True, False]


def test_high_scoring_null_false_positive():
    # FP 0.9 on null, then two TPs: precision envelope is 2/3 everywhere -> AP = 2/3.
    es = EvalSet(
        _images(),
        _dets([("c_null", 0.9, *BOX), ("a_bone", 0.8, *BOX), ("b_bone", 0.7, *BOX)]),
    )
    assert es.ap() == pytest.approx(2 / 3, abs=1e-6)
    # 1 FP over 3 images = 0.33 FPI: nothing qualifies at 0.1, everything at 0.5.
    assert es.recall_at_fpi(0.1) == 0.0
    assert es.recall_at_fpi(0.5) == pytest.approx(1.0)


def test_ap_no_detections_and_no_gt():
    assert average_precision(np.array([]), np.array([], bool), 3) == 0.0
    assert np.isnan(average_precision(np.array([0.5]), np.array([False]), 0))
    assert np.isnan(recall_at_fpi(np.array([]), np.array([], bool), 0, 5, 0.1))


def test_subset_respects_cell_multiplicity():
    es = EvalSet(_images(), _dets([("a_bone", 0.9, *BOX), ("c_null", 0.5, *BOX)]))
    s, tp, n_gt, n_img = es.subset(np.array(["a", "a", "c"]))
    assert n_gt == 2 and n_img == 3
    assert sorted(s.tolist()) == [0.5, 0.9, 0.9]
    assert tp.sum() == 2
    with pytest.raises(KeyError):
        es.subset(np.array(["zzz"]))


def test_paired_bootstrap_of_identical_conditions_is_zero():
    es = EvalSet(_images(), _dets([("a_bone", 0.9, *BOX), ("c_null", 0.95, *BOX)]))
    pt, lo, hi = bootstrap_ci(lambda c: es.ap(c) - es.ap(c), es.cells, n_boot=200)
    assert pt == lo == hi == 0.0


def test_bootstrap_ci_brackets_point():
    es = EvalSet(
        _images(),
        _dets([("c_null", 0.9, *BOX), ("a_bone", 0.8, *BOX), ("b_bone", 0.7, *FAR)]),
    )
    pt, lo, hi = bootstrap_ci(es.ap, es.cells, n_boot=500, seed=1)
    assert lo <= pt <= hi
