"""Turn merged gprMax B-scans into detector images plus YOLO labels.

Pipeline position
-----------------
generate_gprmax_models -> run_gprmax_batch -> **this** -> experiment_03 training.

Labels come from the measured target response, not from travel-time formulas
---------------------------------------------------------------------------
Every scene has a null twin that is identical except the target is replaced by host material
(same clutter, same antenna, same everything). So ``bscan(target) - bscan(null)`` is exactly the
target's own response, including the source-wavelet delay, near-field effects, and the bottom
reflection. The box is the time span where that response is strong, over a lateral span of the
apex +/- ``half_width_depths * depth``. The pilot's analytic box ignored the Ricker delay
(sqrt(2)/f, ~750 samples at 400 MHz) and missed the target entirely, which is why this is
measured instead.

Image variants
--------------
Each variant is a rendering recipe; labels are identical across variants.
  bgr: background removal -> time gain -> percentile clip. Matches the real benchmark's look
       (direct wave suppressed) and is the primary experiment_03 input.
  raw: time gain -> percentile clip. Kept as an ablation; the direct wave sets the clip level
       and squashes deep targets.

B-scan convention here is (n_samples, n_traces): rows are time, columns antenna positions.

Run:
    python -m src.data.process_gprmax_output --config configs/synthetic_controlled.yaml
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.forward_model_1d import C_M_PER_NS
from src.data.preprocess_gpr import remove_background

# Single detection class. Target type (void / bone / anti_bone) is the experimental domain,
# not a class the detector predicts.
ANOMALY_CLASS_ID = 0

DEFAULT_VARIANTS = {
    "bgr": {"background_removal": "mean", "gain_power": 2.0, "clip_percentile": 99.0},
    "raw": {"background_removal": None, "gain_power": 2.0, "clip_percentile": 99.0},
}
DEFAULT_LABELS = {"envelope_threshold": 0.2, "half_width_depths": 1.0}


def time_gain(bscan: np.ndarray, power: float = 2.0) -> np.ndarray:
    """Scale each sample by (t_index / n) ** power, a cheap stand-in for AGC/SEC gain."""
    n = bscan.shape[0]
    gain = (np.arange(n, dtype=float) / max(n - 1, 1)) ** power
    return bscan * gain[:, None]


def to_uint8_image(bscan: np.ndarray, clip_percentile: float = 99.0) -> np.ndarray:
    """Symmetric percentile clip to [-1, 1], then map to uint8 with 0 amplitude at mid-gray."""
    x = bscan.astype(float)
    hi = np.percentile(np.abs(x), clip_percentile)
    if hi <= 0:
        return np.zeros(x.shape, dtype=np.uint8)
    x = np.clip(x / hi, -1.0, 1.0)
    return ((x + 1.0) / 2.0 * 255.0).astype(np.uint8)


def render(bscan: np.ndarray, variant: dict) -> np.ndarray:
    """Apply one variant recipe to a (n_samples, n_traces) B-scan. Returns uint8."""
    x = bscan
    if variant.get("background_removal"):
        x = remove_background(x, trace_axis=1, method=variant["background_removal"])
    x = time_gain(x, float(variant.get("gain_power", 2.0)))
    return to_uint8_image(x, float(variant.get("clip_percentile", 99.0)))


def _host_velocity_m_per_ns(host_eps: float) -> float:
    return C_M_PER_NS / np.sqrt(host_eps)


def apex_trace_index(row: dict) -> float:
    """Fractional trace index where the antenna midpoint is directly over the target."""
    mid0 = (float(row["src_x0_m"]) + float(row["rx_x0_m"])) / 2.0
    return (float(row["target_x_m"]) - mid0) / float(row["trace_step_m"])


def target_response(bscan: np.ndarray, null_bscan: np.ndarray) -> np.ndarray:
    """The target's own response: the paired null twin differs only by the target."""
    if bscan.shape != null_bscan.shape:
        raise ValueError(f"Twin shape mismatch: {bscan.shape} vs null {null_bscan.shape}")
    return bscan - null_bscan


def bbox_from_response(
    response: np.ndarray,
    row: dict,
    envelope_threshold: float = 0.2,
    half_width_depths: float = 1.0,
) -> tuple[int, float, float, float, float]:
    """YOLO box (class, cx, cy, w, h) in [0, 1] from a measured target response.

    Lateral extent: apex +/- half_width_depths * depth (the apex and upper limbs, the part of a
    hyperbola that identifies it). Vertical extent: the samples where the response amplitude,
    maxed over that lateral span, exceeds ``envelope_threshold`` of its peak. Edges are clipped
    to the image before computing center and size, so boxes at a border stay correct.
    """
    n_samples, n_traces = response.shape
    apex = apex_trace_index(row)
    half = max(float(row["depth_m"]) * half_width_depths / float(row["trace_step_m"]), 3.0)
    x0 = max(0.0, apex - half)
    x1 = min(float(n_traces), apex + half)
    if x1 <= x0:
        raise ValueError(f"{row.get('in_file')}: target apex lies outside the survey line")

    span = np.abs(response[:, int(np.floor(x0)) : int(np.ceil(x1))])
    profile = span.max(axis=1)
    peak = profile.max()
    if peak <= 0:
        raise ValueError(f"{row.get('in_file')}: zero target response; is the null twin wrong?")
    rows = np.flatnonzero(profile >= envelope_threshold * peak)
    y0, y1 = float(rows[0]), float(rows[-1] + 1)

    return (
        ANOMALY_CLASS_ID,
        (x0 + x1) / 2.0 / n_traces,
        (y0 + y1) / 2.0 / n_samples,
        (x1 - x0) / n_traces,
        (y1 - y0) / n_samples,
    )


def read_merged_bscan(path: Path, component: str = "Ez") -> tuple[np.ndarray, float]:
    """Read a merged gprMax HDF5 B-scan. Returns (bscan (n_samples, n_traces), dt seconds)."""
    import h5py

    with h5py.File(path, "r") as f:
        dt = float(f.attrs["dt"])
        data = f["rxs"]["rx1"][component][()]
    return np.asarray(data, dtype=float), dt


def merged_path(raw_dir: Path, in_file: str) -> Path:
    return raw_dir / f"{Path(in_file).stem}_merged.out"


def load_manifest(path: Path):
    """Read a manifest CSV into pandas without losing the null class.

    pandas parses the string "null" as NaN by default, which silently drops every null-target
    row from groupbys and filters. Always load manifests through this.
    """
    import pandas as pd

    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def group_by_scene(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """scene_id -> target_type -> manifest row."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        out[r["scene_id"]][r["target_type"]] = r
    return dict(out)


def process_manifest(
    manifest_path: Path,
    raw_dir: Path,
    out_dir: Path,
    variants: dict[str, dict] | None = None,
    labels: dict | None = None,
    image_size: int = 416,
) -> Path:
    """Render every scene in every variant and write YOLO labels.

    Layout:
        out_dir/<variant>/<target_type>/images/<stem>.png
        out_dir/<variant>/<target_type>/labels/<stem>.txt
        out_dir/processed_manifest.csv   (one row per stem; labels shared across variants)

    A scene is skipped (and reported) unless all four twins are merged, because labels need the
    null twin and a partial scene would break the paired design downstream.
    """
    from PIL import Image

    variants = variants or DEFAULT_VARIANTS
    labels = {**DEFAULT_LABELS, **(labels or {})}
    scenes = group_by_scene(list(csv.DictReader(manifest_path.open())))

    processed, skipped = [], []
    for scene_id, twins in sorted(scenes.items()):
        paths = {t: merged_path(raw_dir, r["in_file"]) for t, r in twins.items()}
        if "null" not in twins or not all(p.exists() for p in paths.values()):
            skipped.append(scene_id)
            continue
        null_bscan, _ = read_merged_bscan(paths["null"])

        for ttype, row in twins.items():
            stem = Path(row["in_file"]).stem
            bscan = null_bscan if ttype == "null" else read_merged_bscan(paths[ttype])[0]
            box = None
            if ttype != "null":
                box = bbox_from_response(
                    target_response(bscan, null_bscan),
                    row,
                    labels["envelope_threshold"],
                    labels["half_width_depths"],
                )
            label_txt = "" if box is None else "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*box)

            for vname, vcfg in variants.items():
                base = out_dir / vname / ttype
                (base / "images").mkdir(parents=True, exist_ok=True)
                (base / "labels").mkdir(parents=True, exist_ok=True)
                img = Image.fromarray(render(bscan, vcfg))
                img.resize((image_size, image_size), Image.Resampling.BILINEAR).save(
                    base / "images" / f"{stem}.png"
                )
                (base / "labels" / f"{stem}.txt").write_text(label_txt)

            processed.append(
                {
                    "stem": stem,
                    "scene_id": scene_id,
                    "target_type": ttype,
                    "host_name": row["host_name"],
                    "depth_m": row["depth_m"],
                    "freq_mhz": row["freq_mhz"],
                    "x_center_m": row["x_center_m"],
                    "clutter_realization": row.get("clutter_realization", 0),
                    "has_target": box is not None,
                    "label": label_txt.strip(),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    pm = out_dir / "processed_manifest.csv"
    if processed:
        with pm.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(processed[0].keys()))
            writer.writeheader()
            writer.writerows(processed)
    print(
        f"[process_gprmax_output] {len(processed)} images x {len(variants)} variants "
        f"-> {out_dir}; skipped {len(skipped)} incomplete scenes"
    )
    for s in skipped[:10]:
        print(f"  [skip] {s} (missing a merged twin)")
    return pm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert gprMax B-scans to images + YOLO labels.")
    p.add_argument("--config", type=Path, default=Path("configs/synthetic_controlled.yaml"))
    p.add_argument("--raw-dir", type=Path, default=None, help="Default: synthetic/<exp name>")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: <raw-dir>/images")
    return p.parse_args()


def main() -> None:
    from src.data.generate_gprmax_models import default_out_dir, load_config

    args = parse_args()
    cfg = load_config(args.config)
    raw_dir = args.raw_dir or default_out_dir(cfg)
    proc = cfg.get("processing", {})
    process_manifest(
        raw_dir / "manifest.csv",
        raw_dir,
        args.out_dir or raw_dir / "images",
        variants=proc.get("variants"),
        labels=proc.get("labels"),
        image_size=int(proc.get("image_size", 416)),
    )


if __name__ == "__main__":
    main()
