"""Turn gprMax B-scan output into images plus YOLO labels.

Pipeline position
-----------------
After I run the controlled models from generate_gprmax_models.py through gprMax on the CUDA
box and merge the per-trace outputs, each scene gives one merged HDF5 B-scan. This script
reads those, makes a grayscale image in the same spirit as my real data, and writes a YOLO
label box around the target. The box comes from the known geometry in the manifest, not from
guessing, since this is synthetic data and I placed the target myself.

Null scenes get an empty label (YOLO convention for a background image with no object).

What this does and does not assume
----------------------------------
- It reads the field component Ez, because my source is a z-directed hertzian dipole (2D TMz).
- The B-scan is (n_samples, n_traces): rows are time/depth, columns are antenna positions.
- The bounding box is an approximation of the hyperbola: centered on the apex (antenna over the
  target) and opened out to roughly one target-depth of offset on each side. That is enough to
  localize the anomaly for a detector. It can be tightened later if needed.

The HDF5 read path runs on the CUDA box against real gprMax output. The image and box math are
pure functions and are unit-tested here on synthetic arrays.

Run (on the machine that has the merged .out files):
    python -m src.data.process_gprmax_output \
        --manifest data/processed/synthetic/controlled/manifest.csv \
        --raw-dir  data/processed/synthetic/controlled \
        --out-dir  data/processed/synthetic/images
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.config import PROCESSED_DATA_DIR
from src.data.forward_model_1d import C_M_PER_NS

# Single detection class. The target type (void / bone / anti_bone) is the experimental
# domain, not a class the detector predicts. A detector is trained to find "anomaly".
ANOMALY_CLASS_ID = 0


def time_gain(bscan: np.ndarray, power: float = 2.0) -> np.ndarray:
    """Apply a simple time gain so deeper (later) reflections are visible.

    GPR amplitude falls off fast with depth, so without a gain the shallow direct wave
    dominates and the target is invisible. I scale each sample by (t_index ** power),
    normalized, which is a cheap stand-in for the usual AGC/SEC gain.
    """
    n = bscan.shape[0]
    gain = (np.arange(n, dtype=float) / max(n - 1, 1)) ** power
    return bscan * gain[:, None]


def to_uint8_image(bscan: np.ndarray, clip_percentile: float = 99.0) -> np.ndarray:
    """Normalize a B-scan to a uint8 grayscale image.

    Clips at a high percentile so a few hot samples do not crush the contrast, then maps to
    0-255. Returns an array shaped like the input (n_samples, n_traces).
    """
    x = bscan.astype(float)
    hi = np.percentile(np.abs(x), clip_percentile)
    if hi <= 0:
        return np.zeros(x.shape, dtype=np.uint8)
    x = np.clip(x / hi, -1.0, 1.0)
    x = (x + 1.0) / 2.0  # [-1,1] -> [0,1]
    return (x * 255.0).astype(np.uint8)


def _host_velocity_m_per_ns(host_eps: float) -> float:
    return C_M_PER_NS / np.sqrt(host_eps)


def bbox_yolo(
    row: dict, n_samples: int, dt_s: float
) -> tuple[int, float, float, float, float] | None:
    """Compute a YOLO box (class, cx, cy, w, h), normalized to [0,1], from a manifest row.

    Returns None for the null target (no object). The box is centered on the hyperbola apex
    and opened to about one target-depth of offset on each side.
    """
    if row["target_type"] == "null":
        return None

    depth = float(row["depth_m"])
    host_eps = float(row["host_eps"])
    step = float(row["trace_step_m"])
    n_traces = int(row["n_traces"])
    target_x = float(row["target_x_m"])
    src_x0 = float(row["src_x0_m"])
    rx_x0 = float(row["rx_x0_m"])

    v = _host_velocity_m_per_ns(host_eps)  # m/ns
    dt_ns = dt_s * 1e9

    # Apex: antenna midpoint over the target.
    mid_start = (src_x0 + rx_x0) / 2.0
    apex_trace = (target_x - mid_start) / step
    apex_sample = (2.0 * depth / v) / dt_ns  # twt at the apex, in samples

    # Open the box to ~one depth of offset on each side (hyperbola limbs).
    half_off_m = depth
    half_traces = max(half_off_m / step, 3.0)
    # Bottom of the limbs at that offset.
    t_limb_ns = 2.0 * np.sqrt(depth**2 + half_off_m**2) / v
    limb_sample = t_limb_ns / dt_ns
    pad = max(0.04 * n_samples, 5.0)  # a little vertical padding for the wavelet

    x_left = apex_trace - half_traces
    x_right = apex_trace + half_traces
    y_top = apex_sample - pad
    y_bottom = limb_sample + pad

    cx = (x_left + x_right) / 2.0 / n_traces
    cy = (y_top + y_bottom) / 2.0 / n_samples
    w = (x_right - x_left) / n_traces
    h = (y_bottom - y_top) / n_samples

    # Clamp to the image.
    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))
    w = float(np.clip(w, 0.0, 1.0))
    h = float(np.clip(h, 0.0, 1.0))
    return (ANOMALY_CLASS_ID, cx, cy, w, h)


def read_merged_bscan(path: Path, component: str = "Ez") -> tuple[np.ndarray, float]:
    """Read a merged gprMax HDF5 B-scan. Returns (bscan (n_samples, n_traces), dt seconds).

    Only imported h5py here so the rest of the module imports without it.
    """
    import h5py

    with h5py.File(path, "r") as f:
        dt = float(f.attrs["dt"])
        data = f["rxs"]["rx1"][component][()]
    return np.asarray(data, dtype=float), dt


def process_manifest(
    manifest_path: Path,
    raw_dir: Path,
    out_dir: Path,
    image_size: int = 416,
) -> Path:
    """Convert every merged .out referenced by the manifest into an image + YOLO label.

    Output layout (organized by target type so the experiment can pick domains per condition):
        out_dir/<target_type>/images/<stem>.png
        out_dir/<target_type>/labels/<stem>.txt
        out_dir/processed_manifest.csv
    """
    from PIL import Image

    rows = list(csv.DictReader(manifest_path.open()))
    processed = []
    for row in rows:
        stem = Path(row["in_file"]).stem
        merged = raw_dir / f"{stem}_merged.out"
        if not merged.exists():
            print(f"[skip] missing {merged.name} (run gprMax + merge first)")
            continue

        bscan, dt = read_merged_bscan(merged)
        img = to_uint8_image(time_gain(bscan))
        pil = Image.fromarray(img).resize((image_size, image_size), Image.Resampling.BILINEAR)

        ttype = row["target_type"]
        img_dir = out_dir / ttype / "images"
        lbl_dir = out_dir / ttype / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        pil.save(img_dir / f"{stem}.png")
        box = bbox_yolo(row, n_samples=bscan.shape[0], dt_s=dt)
        label_path = lbl_dir / f"{stem}.txt"
        if box is None:
            label_path.write_text("")  # background image
        else:
            cls, cx, cy, w, h = box
            label_path.write_text(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        processed.append(
            {
                "scene_id": row["scene_id"],
                "target_type": ttype,
                "image": str((img_dir / f"{stem}.png").relative_to(out_dir)),
                "label": str(label_path.relative_to(out_dir)),
                "has_target": box is not None,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pm = out_dir / "processed_manifest.csv"
    if processed:
        with pm.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(processed[0].keys()))
            writer.writeheader()
            writer.writerows(processed)
    print(f"[process_gprmax_output] processed {len(processed)} scenes -> {out_dir}")
    return pm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert gprMax B-scans to images + YOLO labels.")
    base = PROCESSED_DATA_DIR / "synthetic" / "controlled"
    p.add_argument("--manifest", type=Path, default=base / "manifest.csv")
    p.add_argument("--raw-dir", type=Path, default=base)
    p.add_argument("--out-dir", type=Path, default=PROCESSED_DATA_DIR / "synthetic" / "images")
    p.add_argument("--image-size", type=int, default=416)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    process_manifest(args.manifest, args.raw_dir, args.out_dir, args.image_size)


if __name__ == "__main__":
    main()
