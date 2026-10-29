"""experiment_03 C0-C2 + Cnull: train YOLO detectors per condition and fold, then analyze.

Design (docs/experiments/experiment_03_transfer.md, amendments A1/A2)
--------------------------------------------------------------------
- Unit of independence is the scene cell. Grouped, stratified K-fold over cells: each run trains
  on k-2 folds, picks its checkpoint on 1 val fold, and predicts on the held-out test fold for
  all four target types. Pooled out-of-fold predictions cover every cell exactly once per seed.
- Checkpoint selection only ever sees the training domain (void runs never validate on bone).
- Null twins of the same cells are included as background images in train, val, and eval.
- Metrics and paired bootstrap CIs over cells come from src.models.detection_metrics.

Runs are resumable: a run with a DONE.json is skipped.

    python -m src.models.transfer_experiment run     --config configs/experiment_03_transfer.yaml
    python -m src.models.transfer_experiment analyze --config configs/experiment_03_transfer.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import ROOT_DIR
from src.data.generate_gprmax_models import default_out_dir, load_config
from src.data.process_gprmax_output import load_manifest
from src.models.detection_metrics import EvalSet, bootstrap_ci

TARGET_TYPES = ("void", "bone", "anti_bone", "null")


# --------------------------------------------------------------------------- data tables


def parse_label(text: str) -> tuple[float, float, float, float] | None:
    parts = str(text).split()
    if not parts:
        return None
    if len(parts) != 5:
        raise ValueError(f"Bad YOLO label: {text!r}")
    return tuple(float(v) for v in parts[1:])  # type: ignore[return-value]


def synthetic_images(images_dir: Path, variant: str) -> pd.DataFrame:
    """One row per synthetic image: paths, cell, target type, strata, and GT box."""
    pm = load_manifest(images_dir / "processed_manifest.csv")
    base = images_dir / variant
    df = pd.DataFrame(
        {
            "image": pm["stem"],
            "cell": pm["scene_id"],
            "target_type": pm["target_type"],
            "host_name": pm["host_name"],
            "freq_mhz": pm["freq_mhz"].astype(float),
            "depth_m": pm["depth_m"].astype(float),
            "img_path": [base / t / "images" / f"{s}.png" for s, t in zip(pm.stem, pm.target_type)],
            "lbl_path": [base / t / "labels" / f"{s}.txt" for s, t in zip(pm.stem, pm.target_type)],
        }
    )
    df["gt"] = [parse_label(p.read_text()) for p in df["lbl_path"]]
    missing = [p for p in df["img_path"] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[0]}")
    if (df["gt"].isna() != (df["target_type"] == "null")).any():
        raise ValueError("Label/target-type mismatch: only null images may be unlabelled")
    return df


def real_images(detection_dir: Path, split: str) -> pd.DataFrame:
    imgs = sorted(p for p in (detection_dir / split / "images").iterdir() if p.is_file())
    return pd.DataFrame(
        {
            "image": [f"real_{p.stem}" for p in imgs],
            "img_path": imgs,
            "lbl_path": [detection_dir / split / "labels" / f"{p.stem}.txt" for p in imgs],
        }
    )


def make_folds(cells: pd.DataFrame, k: int, strata: list[str], seed: int) -> pd.Series:
    """Assign each cell to a fold, balanced within every stratum. Returns cell -> fold."""
    rng = np.random.default_rng(seed)
    assignment: dict[str, int] = {}
    offset = 0
    for _, grp in cells.groupby(strata, sort=True):
        ids = grp["cell"].to_numpy()[rng.permutation(len(grp))]
        for i, c in enumerate(ids):
            assignment[c] = (offset + i) % k
        offset += len(ids)  # continue round-robin so fold totals stay balanced across strata
    return pd.Series(assignment, name="fold")


def fold_roles(test_fold: int, k: int) -> tuple[list[int], int]:
    """Train folds and val fold for a given test fold (val = the next fold, cyclically)."""
    val = (test_fold + 1) % k
    return [f for f in range(k) if f not in (test_fold, val)], val


# --------------------------------------------------------------------------- one run


def _link_split(rows: pd.DataFrame, split_dir: Path) -> None:
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    for r in rows.itertuples():
        ext = Path(r.img_path).suffix
        for src, dst in (
            (Path(r.img_path), split_dir / "images" / f"{r.image}{ext}"),
            (Path(r.lbl_path), split_dir / "labels" / f"{r.image}.txt"),
        ):
            if not dst.exists():
                dst.symlink_to(src.resolve())


def build_dataset(run_dir: Path, train: pd.DataFrame, val: pd.DataFrame) -> Path:
    """Materialize a YOLO dataset of symlinks (no copies). Returns dataset.yaml."""
    data = run_dir / "data"
    _link_split(train, data / "train")
    _link_split(val, data / "val")
    yml = data / "dataset.yaml"
    yml.write_text(
        yaml.safe_dump(
            {
                "path": str(data),
                "train": "train/images",
                "val": "val/images",
                "names": {0: "anomaly"},
            }
        )
    )
    return yml


def train_and_predict(
    run_dir: Path, dataset_yaml: Path, test: pd.DataFrame, cfg: dict, seed: int
) -> pd.DataFrame:
    from ultralytics import YOLO

    m = cfg["model"]
    model = YOLO(str(ROOT_DIR / m["weights"]))
    model.train(
        data=str(dataset_yaml),
        imgsz=m["imgsz"],
        epochs=m["epochs"],
        patience=m["patience"],
        batch=m["batch"],
        workers=m["workers"],
        device=0,
        seed=seed,
        deterministic=True,
        project=str(run_dir),
        name="train",
        exist_ok=True,
        plots=False,
        verbose=False,
        **m["augment"],
    )
    weights = run_dir / "train" / "weights"
    best = weights / "best.pt" if (weights / "best.pt").exists() else weights / "last.pt"

    return predict(YOLO(str(best)), test, cfg)


def predict(model, images: pd.DataFrame, cfg: dict, batch: int = 32) -> pd.DataFrame:
    """Score every image. Results are paired to inputs by position, never by r.path, because
    ultralytics reports list sources as image0, image1, ... rather than file names."""
    p, imgsz = cfg["predict"], cfg["model"]["imgsz"]
    rows = []
    for i in range(0, len(images), batch):
        chunk = images.iloc[i : i + batch]
        results = model.predict(
            source=[str(x) for x in chunk["img_path"]],
            imgsz=imgsz,
            conf=p["conf"],
            iou=p["iou"],
            max_det=p["max_det"],
            verbose=False,
        )
        if len(results) != len(chunk):
            raise RuntimeError(f"predict returned {len(results)} results for {len(chunk)} images")
        for name, r in zip(chunk["image"], results):
            for (cx, cy, w, h), s in zip(r.boxes.xywhn.tolist(), r.boxes.conf.tolist()):
                rows.append((name, s, cx, cy, w, h))
    return pd.DataFrame(rows, columns=["image", "score", "cx", "cy", "w", "h"])


# --------------------------------------------------------------------------- orchestration


class Experiment:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        syn_cfg = load_config(ROOT_DIR / cfg["data"]["synthetic_config"])
        self.variant = cfg["data"]["variant"]
        self.images = synthetic_images(default_out_dir(syn_cfg) / "images", self.variant)
        cells = self.images.drop_duplicates("cell")[["cell", *cfg["cv"]["stratify_by"]]]
        self.k = int(cfg["cv"]["k"])
        self.folds = make_folds(cells, self.k, cfg["cv"]["stratify_by"], cfg["experiment"]["seed"])
        self.images["fold"] = self.images["cell"].map(self.folds)
        self.root = ROOT_DIR / cfg["experiment"]["results_dir"] / self.variant

    def run_dir(self, cond: str, fold: int, seed_idx: int) -> Path:
        return self.root / cond / f"fold{fold}_s{seed_idx}"

    def run_all(self, only: list[str] | None = None) -> None:
        conds = self.cfg["train_conditions"]
        for cond in only or list(conds):
            for seed_idx in range(int(self.cfg["cv"]["seeds_per_fold"])):
                for fold in range(self.k):
                    self.run_one(cond, conds[cond], fold, seed_idx)

    def run_one(self, cond: str, spec: dict, fold: int, seed_idx: int) -> None:
        rd = self.run_dir(cond, fold, seed_idx)
        if (rd / "DONE.json").exists():
            print(f"[skip] {cond} fold{fold} s{seed_idx}", flush=True)
            return
        t0 = time.time()
        train_folds, val_fold = fold_roles(fold, self.k)
        types = [*spec["target_types"], "null"]
        im = self.images
        train = im[im["fold"].isin(train_folds) & im["target_type"].isin(types)]
        val = im[(im["fold"] == val_fold) & im["target_type"].isin(types)]
        test = im[im["fold"] == fold]  # all four target types
        if spec.get("real"):
            train = pd.concat(
                [train, real_images(ROOT_DIR / self.cfg["data"]["real_detection_dir"], "train")]
            )
        assert not set(train["image"]) & set(test["image"]), "train/test leakage"

        seed = int(self.cfg["experiment"]["seed"]) + 100 * seed_idx + fold
        rd.mkdir(parents=True, exist_ok=True)
        dets = train_and_predict(rd, build_dataset(rd, train, val), test, self.cfg, seed)
        dets.to_csv(rd / "predictions.csv", index=False)
        summary = {
            "condition": cond,
            "fold": fold,
            "seed_idx": seed_idx,
            "seed": seed,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_dets": len(dets),
            "minutes": round((time.time() - t0) / 60, 2),
        }
        (rd / "DONE.json").write_text(json.dumps(summary, indent=1))
        print(f"[done] {summary}", flush=True)

    # ------------------------------------------------------------------ analysis

    def eval_sets(self, cond: str, eval_type: str) -> list[EvalSet]:
        """One EvalSet per seed replicate: pooled out-of-fold dets on eval_type + null images."""
        imgs = self.images[self.images["target_type"].isin([eval_type, "null"])]
        out = []
        for seed_idx in range(int(self.cfg["cv"]["seeds_per_fold"])):
            parts = []
            for fold in range(self.k):
                f = self.run_dir(cond, fold, seed_idx) / "predictions.csv"
                if not (f.parent / "DONE.json").exists():
                    raise FileNotFoundError(f"missing run {f.parent}; run the experiment first")
                parts.append(pd.read_csv(f))
            out.append(EvalSet(imgs[["image", "cell", "gt"]], pd.concat(parts), self._iou()))
        return out

    def _iou(self) -> float:
        return float(self.cfg["analysis"]["iou_thr"])

    def analyze(self) -> dict:
        a = self.cfg["analysis"]
        n_boot, fpi = int(a["n_boot"]), float(a["fpi"])
        conds = list(self.cfg["train_conditions"])
        es = {(c, t): self.eval_sets(c, t) for c in conds for t in ("void", "bone", "anti_bone")}
        cells = es[(conds[0], "bone")][0].cells

        def ap(c, t):
            return lambda s: float(np.mean([e.ap(s) for e in es[(c, t)]]))

        def rec(c, t):
            return lambda s: float(np.mean([e.recall_at_fpi(fpi, s) for e in es[(c, t)]]))

        rows = []
        for c in conds:
            for t in ("void", "bone", "anti_bone"):
                m, lo, hi = bootstrap_ci(ap(c, t), cells, n_boot)
                r, rlo, rhi = bootstrap_ci(rec(c, t), cells, n_boot // 5)
                rows.append(
                    {
                        "train": c,
                        "eval": t,
                        "AP50": m,
                        "AP50_lo": lo,
                        "AP50_hi": hi,
                        f"recall@{fpi}fpi": r,
                        "recall_lo": rlo,
                        "recall_hi": rhi,
                    }
                )
        summary = pd.DataFrame(rows)

        def diff(f1, f2):
            return bootstrap_ci(lambda s: f1(s) - f2(s), cells, n_boot)

        decisions = {}
        if {"bone", "real_void"} <= set(conds):
            decisions["H1a"] = _decide(*diff(ap("bone", "bone"), ap("real_void", "bone")))
        if "void" in conds:
            decisions["H1b"] = _decide(*diff(ap("void", "anti_bone"), ap("void", "bone")))
        if {"bone", "void"} <= set(conds):
            # Exploratory (not pre-registered): the gap without real data in the source domain.
            decisions["exploratory_syn_gap"] = _decide(
                *diff(ap("bone", "bone"), ap("void", "bone"))
            )

        per_stratum = self._per_stratum(conds, es, max(n_boot // 5, 1000))
        extras = self._polarity_and_null(float(a["report_conf"]))

        out = self.root / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "summary.csv", index=False)
        per_stratum.to_csv(out / "per_stratum.csv", index=False)
        (out / "decisions.json").write_text(json.dumps({**decisions, **extras}, indent=1))
        return {"summary": summary, "per_stratum": per_stratum, "decisions": decisions, **extras}

    def _per_stratum(self, conds, es, n_boot) -> pd.DataFrame:
        strata = self.images.drop_duplicates("cell").groupby(["host_name", "freq_mhz"])["cell"]
        rows = []
        for (host, freq), cells in strata:
            cells = np.sort(cells.to_numpy())
            for c in conds:
                for t in ("void", "bone", "anti_bone"):
                    stat = lambda s, c=c, t=t: float(np.mean([e.ap(s) for e in es[(c, t)]]))  # noqa: E731
                    m, lo, hi = bootstrap_ci(stat, cells, n_boot)
                    rows.append(
                        {
                            "host": host,
                            "freq_mhz": freq,
                            "train": c,
                            "eval": t,
                            "AP50": m,
                            "AP50_lo": lo,
                            "AP50_hi": hi,
                        }
                    )
        return pd.DataFrame(rows)

    def _polarity_and_null(self, conf: float) -> dict:
        """Polarity-confusion detection rates (void model) and Cnull false positives per image."""
        if "void" not in self.cfg["train_conditions"]:
            return {}
        rates = {}
        for t in ("void", "bone", "anti_bone", "null"):
            vals = []
            for e in self.eval_sets("void", "bone" if t == "null" else t):
                d = e.dets[e.dets["score"] >= conf]
                imgs = e.images[e.images["image"].str.endswith(f"__{t}")]
                if t == "null":
                    vals.append(d["image"].isin(imgs["image"]).sum() / len(imgs))  # FP per image
                else:
                    hit = d[d["tp"] & d["image"].isin(imgs["image"])]["image"].nunique()
                    vals.append(hit / len(imgs))
            rates[t] = float(np.mean(vals))
        return {
            f"void_model_detection_rate@{conf}": {
                k: rates[k] for k in ("void", "bone", "anti_bone")
            },
            f"polarity_confusion@{conf}": rates["anti_bone"] - rates["bone"],
            f"Cnull_fp_per_image@{conf}": rates["null"],
        }


def _decide(point: float, lo: float, hi: float) -> dict:
    """Pre-registered rule: supported if >= 0.10 and CI excludes 0; refuted if < 0.05 or CI
    includes 0; otherwise inconclusive."""
    excl0 = lo > 0 or hi < 0
    if point >= 0.10 and excl0:
        verdict = "supported"
    elif point < 0.05 or not excl0:
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    return {"diff": point, "ci_lo": lo, "ci_hi": hi, "verdict": verdict}


def main() -> None:
    p = argparse.ArgumentParser(description="experiment_03 C0-C2 transfer study")
    p.add_argument("action", choices=["run", "analyze", "folds"])
    p.add_argument("--config", type=Path, default=Path("configs/experiment_03_transfer.yaml"))
    p.add_argument("--only", nargs="*", help="train conditions to run (default: all)")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    exp = Experiment(cfg)
    if args.action == "folds":
        cells = exp.images.drop_duplicates("cell")
        print(cells.groupby(["fold", "host_name", "freq_mhz"]).size().unstack([1, 2]))
    elif args.action == "run":
        exp.run_all(args.only)
    else:
        res = exp.analyze()
        pd.set_option("display.width", 200)
        print(res["summary"].round(3).to_string(index=False))
        print(
            json.dumps(
                {k: v for k, v in res.items() if k not in ("summary", "per_stratum")},
                indent=1,
                default=float,
            )
        )


if __name__ == "__main__":
    main()
