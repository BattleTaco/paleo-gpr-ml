"""Detection metrics and scene-cell bootstrap for experiment_03 (sec 4 and 6).

Why not ultralytics' val(): the experiment pools out-of-fold predictions across runs and needs
bootstrap CIs that resample *scene cells* (the independent unit, since twins and variants of a
cell are correlated) and *paired* differences between conditions evaluated on the same cells.
That needs the raw scored detections, so the metrics live here.

Conventions
-----------
- Single class. An image has at most one ground-truth box (None for null images).
- Boxes are normalized YOLO (cx, cy, w, h).
- A detection is a TP if it is the highest-scoring unmatched detection with IoU >= iou_thr to
  the image's GT; any other detection is an FP (greedy matching, COCO style).
- AP uses COCO 101-point interpolated precision.

Detections are a DataFrame with columns: image, score, cx, cy, w, h.
Images are a DataFrame with columns: image, cell, gt (tuple or None), plus any metadata.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N, 4) and a single box b (4,), all normalized cx, cy, w, h."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    ax0, ay0 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2
    ax1, ay1 = a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None)
    ih = np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None)
    inter = iw * ih
    union = a[:, 2] * a[:, 3] + b[2] * b[3] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_detections(
    images: pd.DataFrame, dets: pd.DataFrame, iou_thr: float = 0.5
) -> pd.DataFrame:
    """Label every detection TP/FP. Returns dets with an added boolean ``tp`` column.

    Only images listed in ``images`` are scored; detections on other images are dropped.
    """
    gts = dict(zip(images["image"], images["gt"]))
    out = dets[dets["image"].isin(gts)].copy()
    out["tp"] = False
    for img, grp in out.groupby("image", sort=False):
        gt = gts[img]
        if gt is None or len(grp) == 0:
            continue
        order = grp["score"].to_numpy().argsort()[::-1]
        ious = iou_xywh(grp[["cx", "cy", "w", "h"]].to_numpy()[order], np.asarray(gt, float))
        hit = np.flatnonzero(ious >= iou_thr)
        if hit.size:
            out.loc[grp.index[order[hit[0]]], "tp"] = True  # highest-scoring match only
    return out


def average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """COCO 101-point interpolated AP from scored detections already labelled TP/FP."""
    if n_gt == 0:
        return float("nan")
    if scores.size == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp[order].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1.0 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    # Monotone precision envelope, then sample at 101 recall points.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    pts = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, pts, side="left")
    return float(np.mean([precision[i] if i < precision.size else 0.0 for i in idx]))


def recall_at_fpi(
    scores: np.ndarray, tp: np.ndarray, n_gt: int, n_images: int, fpi: float
) -> float:
    """Max recall over thresholds whose false positives per image stay <= fpi."""
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    tp = tp[order].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1.0 - tp)
    ok = cfp / max(n_images, 1) <= fpi
    return float(ctp[ok].max() / n_gt) if ok.any() else 0.0


class EvalSet:
    """Matched detections for one (model, evaluation set), precomputed for fast resampling.

    Resampling is done by integer cell codes and np.bincount, so a 10k bootstrap is cheap.
    """

    def __init__(self, images: pd.DataFrame, dets: pd.DataFrame, iou_thr: float = 0.5):
        self.images = images.reset_index(drop=True)
        self.dets = match_detections(self.images, dets, iou_thr)
        self.cells = np.sort(self.images["cell"].unique())
        img_code = np.searchsorted(self.cells, self.images["cell"].to_numpy())
        code_by_img = dict(zip(self.images["image"], img_code))
        self._img_code = img_code
        self._img_has_gt = self.images["gt"].notna().to_numpy()
        self._det_code = self.dets["image"].map(code_by_img).to_numpy(dtype=int)
        self._scores = self.dets["score"].to_numpy(dtype=float)
        self._tp = self.dets["tp"].to_numpy(dtype=bool)

    def subset(self, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Scores, tp flags, n_gt, n_images for a multiset of cells (repeats allowed)."""
        codes = np.searchsorted(self.cells, np.asarray(cells))
        if codes.size and (
            codes.max() >= self.cells.size or not np.all(self.cells[codes] == cells)
        ):
            raise KeyError("cells not in this evaluation set")
        counts = np.bincount(codes, minlength=self.cells.size)
        img_w = counts[self._img_code]
        det_w = counts[self._det_code]
        return (
            np.repeat(self._scores, det_w),
            np.repeat(self._tp, det_w),
            int((img_w * self._img_has_gt).sum()),
            int(img_w.sum()),
        )

    def ap(self, cells: np.ndarray | None = None) -> float:
        s, t, n_gt, _ = self.subset(self.cells if cells is None else cells)
        return average_precision(s, t, n_gt)

    def recall_at_fpi(self, fpi: float, cells: np.ndarray | None = None) -> float:
        s, t, n_gt, n_img = self.subset(self.cells if cells is None else cells)
        return recall_at_fpi(s, t, n_gt, n_img, fpi)


def bootstrap_ci(
    stat: Callable[[np.ndarray], float],
    cells: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI, resampling scene cells with replacement.

    ``stat`` takes a multiset of cells. For a paired difference, pass a stat that evaluates
    both conditions on the same multiset, which is what makes the comparison paired.
    """
    rng = np.random.default_rng(seed)
    cells = np.asarray(cells)
    point = stat(cells)
    boots = np.array(
        [stat(rng.choice(cells, size=cells.size, replace=True)) for _ in range(n_boot)]
    )
    boots = boots[np.isfinite(boots)]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2]) if boots.size else (np.nan, np.nan)
    return float(point), float(lo), float(hi)
