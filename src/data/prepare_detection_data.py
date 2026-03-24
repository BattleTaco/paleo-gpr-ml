"""Prepare GPR B-scan data in YOLO detection format.

Reorganizes the augmented images and YOLO annotations into the directory
structure expected by Ultralytics YOLOv8:

    data/processed/detection/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── dataset.yaml

For intact images (no annotations), label files are empty (YOLO convention
for negative/background images).

Supports two framing modes:
    - "2class": cavity (0) and utility (1) as separate classes
    - "binary": both collapsed to anomaly (0)

Usage:
    python -m src.data.prepare_detection_data [--mode 2class] [--seed 42]
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data.build_splits import (
    collect_images_by_original_id,
    stratified_id_split,
    verify_no_leakage,
    CLASS_DIRS,
    IMG_EXTENSIONS,
)


# YOLO annotation source directories
ANNOTATION_DIRS = {
    "cavities": RAW_DATA_DIR / "GPR_data" / "augmented_cavities" / "annotations" / "Yolo_format",
    "utilities": RAW_DATA_DIR / "GPR_data" / "augmented_utilities" / "annotations" / "YOLO_format",
}

# Class mapping for detection
CLASS_MAP_2CLASS = {"cavities": 0, "utilities": 1}
CLASS_MAP_BINARY = {"cavities": 0, "utilities": 0}


def prepare_detection_data(
    mode: str = "2class",
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Path:
    """Build detection dataset in YOLO format.

    Args:
        mode: "2class" (cavity vs utility) or "binary" (anomaly vs background)
        seed: random seed for splitting
        train_ratio: fraction for training
        val_ratio: fraction for validation

    Returns:
        Path to dataset.yaml
    """
    class_map = CLASS_MAP_2CLASS if mode == "2class" else CLASS_MAP_BINARY
    class_names = ["cavity", "utility"] if mode == "2class" else ["anomaly"]

    output_dir = PROCESSED_DATA_DIR / "detection" / mode
    print(f"Preparing detection data: mode={mode}, output={output_dir}")

    # Reuse the same ID-based split logic from classification
    grouped = collect_images_by_original_id(CLASS_DIRS)
    splits = stratified_id_split(grouped, train_ratio, val_ratio, seed)
    verify_no_leakage(splits)

    stats = {}

    for split_name, items in splits.items():
        img_dir = output_dir / split_name / "images"
        lbl_dir = output_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        n_positive = 0
        n_negative = 0

        for cls_name, orig_id, img_path in items:
            # Copy image
            dest_img = img_dir / img_path.name
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)

            # Create label file
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            ann_dir = ANNOTATION_DIRS.get(cls_name)

            if ann_dir is not None:
                src_lbl = ann_dir / (img_path.stem + ".txt")
                if src_lbl.exists():
                    # Remap class IDs
                    new_class_id = class_map[cls_name]
                    lines = []
                    with open(src_lbl) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                # Replace original class ID with our mapping
                                parts[0] = str(new_class_id)
                                lines.append(" ".join(parts))

                    with open(lbl_path, "w") as f:
                        f.write("\n".join(lines) + "\n" if lines else "")

                    n_positive += 1
                else:
                    # No annotation file found, write empty label
                    lbl_path.touch()
                    n_negative += 1
            else:
                # Intact class: empty label file (background)
                lbl_path.touch()
                n_negative += 1

        stats[split_name] = {"positive": n_positive, "negative": n_negative, "total": n_positive + n_negative}
        print(f"  {split_name}: {n_positive} positive, {n_negative} negative ({n_positive + n_negative} total)")

    # Write dataset.yaml for YOLOv8
    dataset_config = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(dataset_config, f, default_flow_style=False)

    print(f"\nDataset config written to {yaml_path}")
    print(f"Classes: {class_names}")

    return yaml_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare detection data in YOLO format")
    parser.add_argument("--mode", choices=["2class", "binary"], default="2class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    prepare_detection_data(
        mode=args.mode,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
