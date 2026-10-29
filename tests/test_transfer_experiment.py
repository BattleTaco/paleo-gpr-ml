"""Fold assignment, decision rules, and dataset materialization for experiment_03."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from src.models.transfer_experiment import (
    _decide,
    build_dataset,
    fold_roles,
    make_folds,
    parse_label,
)


def _cells():
    rows = []
    for host in ("dry_sand", "limestone"):
        for freq in (400.0, 800.0):
            for i in range(18):
                rows.append(
                    {"cell": f"{host}_{int(freq)}_{i:02d}", "host_name": host, "freq_mhz": freq}
                )
    return pd.DataFrame(rows)


def test_folds_cover_every_cell_once_and_balance_strata():
    cells = _cells()
    folds = make_folds(cells, 6, ["host_name", "freq_mhz"], seed=1)
    assert set(folds.index) == set(cells["cell"])
    merged = cells.assign(fold=cells["cell"].map(folds))
    # 18 cells per stratum over 6 folds -> exactly 3 per (fold, stratum).
    counts = merged.groupby(["fold", "host_name", "freq_mhz"]).size()
    assert counts.min() == counts.max() == 3


def test_folds_are_deterministic_and_seed_dependent():
    cells = _cells()
    a = make_folds(cells, 6, ["host_name", "freq_mhz"], seed=1)
    assert a.equals(make_folds(cells, 6, ["host_name", "freq_mhz"], seed=1))
    assert not a.equals(make_folds(cells, 6, ["host_name", "freq_mhz"], seed=2))


@pytest.mark.parametrize("test_fold", range(6))
def test_fold_roles_disjoint_and_complete(test_fold):
    train, val = fold_roles(test_fold, 6)
    assert val != test_fold and val not in train and test_fold not in train
    assert sorted([*train, val, test_fold]) == list(range(6))


def test_every_cell_tested_exactly_once_across_runs():
    seen = [f for f in range(6) for _ in [fold_roles(f, 6)]]
    assert sorted(seen) == list(range(6))


@pytest.mark.parametrize(
    "point,lo,hi,verdict",
    [
        (0.20, 0.08, 0.30, "supported"),
        (0.20, -0.01, 0.40, "refuted"),  # CI includes 0
        (0.03, 0.01, 0.05, "refuted"),  # below 0.05
        (0.07, 0.02, 0.12, "inconclusive"),  # between thresholds, CI excludes 0
        (-0.20, -0.30, -0.10, "refuted"),  # effect in the wrong direction
    ],
)
def test_decision_rule_matches_preregistration(point, lo, hi, verdict):
    assert _decide(point, lo, hi)["verdict"] == verdict


def test_parse_label():
    assert parse_label("") is None
    assert parse_label("0 0.5 0.4 0.2 0.1\n") == (0.5, 0.4, 0.2, 0.1)
    with pytest.raises(ValueError):
        parse_label("0 0.5 0.4")


def test_build_dataset_links_images_and_labels(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    rows = []
    for name in ("a__bone", "a__null"):
        (src / f"{name}.png").write_bytes(b"png")
        (src / f"{name}.txt").write_text("0 0.5 0.5 0.1 0.1\n" if "bone" in name else "")
        rows.append(
            {"image": name, "img_path": src / f"{name}.png", "lbl_path": src / f"{name}.txt"}
        )
    df = pd.DataFrame(rows)
    yml = build_dataset(tmp_path / "run", df, df.iloc[:1])
    spec = yaml.safe_load(yml.read_text())
    root = tmp_path / "run" / "data"
    assert spec["names"] == {0: "anomaly"} and spec["path"] == str(root)
    # YOLO finds labels by swapping /images/ for /labels/ in the image path.
    assert (root / "train" / "images" / "a__bone.png").is_symlink()
    assert (root / "train" / "labels" / "a__bone.txt").read_text().startswith("0 ")
    assert (root / "train" / "labels" / "a__null.txt").read_text() == ""
    assert len(list((root / "val" / "images").iterdir())) == 1
