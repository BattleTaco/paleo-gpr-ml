"""Build train/val/test splits by original image ID to prevent data leakage.

Augmented images follow the naming pattern: {original_id}_aug_{n}.jpg
All variants of the same original MUST land in the same split.

Usage:
    python -m src.data.build_splits [--seed 42] [--train-ratio 0.70] [--val-ratio 0.15]

Outputs a split manifest as YAML to data/processed/split_manifest.yaml
and copies/symlinks images into data/processed/{train,val,test}/{class}/ structure.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


# Class directory mapping
CLASS_DIRS = {
    "intact": RAW_DATA_DIR / "GPR_data" / "augmented_intact",
    "cavities": RAW_DATA_DIR / "GPR_data" / "augmented_cavities",
    "utilities": RAW_DATA_DIR / "GPR_data" / "augmented_utilities",
}

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def extract_original_id(filename: str) -> str | None:
    """Extract the original image ID from an augmented filename.

    Expected pattern: {id}_aug_{n}.jpg -> returns id
    """
    match = re.match(r"(\d+)_aug_\d+", Path(filename).stem)
    if match:
        return match.group(1)
    return None


def collect_images_by_original_id(
    class_dirs: dict[str, Path],
) -> dict[str, dict[str, list[Path]]]:
    """Group image paths by (class, original_id).

    Returns:
        {class_name: {original_id: [list of image paths]}}
    """
    grouped: dict[str, dict[str, list[Path]]] = {}

    for cls_name, cls_dir in class_dirs.items():
        grouped[cls_name] = defaultdict(list)
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() not in IMG_EXTENSIONS:
                continue
            orig_id = extract_original_id(f.name)
            if orig_id is None:
                orig_id = f.stem  # fallback: treat entire filename as ID
            grouped[cls_name][orig_id].append(f)

    return grouped


def stratified_id_split(
    grouped: dict[str, dict[str, list[Path]]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[tuple[str, str, Path]]]:
    """Split original IDs into train/val/test, stratified by class.

    Returns:
        {"train": [(class, orig_id, img_path), ...], "val": [...], "test": [...]}
    """
    rng = np.random.RandomState(seed)
    splits: dict[str, list[tuple[str, str, Path]]] = {
        "train": [], "val": [], "test": [],
    }

    for cls_name, id_to_paths in grouped.items():
        orig_ids = sorted(id_to_paths.keys())
        rng.shuffle(orig_ids)

        n = len(orig_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_ids = orig_ids[:n_train]
        val_ids = orig_ids[n_train:n_train + n_val]
        test_ids = orig_ids[n_train + n_val:]

        for split_name, id_list in [
            ("train", train_ids), ("val", val_ids), ("test", test_ids),
        ]:
            for oid in id_list:
                for img_path in id_to_paths[oid]:
                    splits[split_name].append((cls_name, oid, img_path))

    return splits


def write_splits(
    splits: dict[str, list[tuple[str, str, Path]]],
    output_dir: Path,
    copy_files: bool = True,
) -> dict:
    """Write split files to output directory and return manifest.

    Creates:
        output_dir/{train,val,test}/{class_name}/image.jpg
        output_dir/split_manifest.yaml
    """
    manifest = {"split_counts": {}, "split_details": {}}

    for split_name, items in splits.items():
        split_dir = output_dir / split_name
        class_counts: dict[str, int] = defaultdict(int)
        id_sets: dict[str, set] = defaultdict(set)
        file_list = []

        for cls_name, orig_id, img_path in items:
            dest_dir = split_dir / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / img_path.name

            if copy_files and not dest.exists():
                shutil.copy2(img_path, dest)

            class_counts[cls_name] += 1
            id_sets[cls_name].add(orig_id)
            file_list.append(str(img_path.relative_to(img_path.parents[4])))

        manifest["split_counts"][split_name] = {
            "total": len(items),
            "per_class": dict(class_counts),
            "unique_ids_per_class": {k: len(v) for k, v in id_sets.items()},
        }

    return manifest


def verify_no_leakage(
    splits: dict[str, list[tuple[str, str, Path]]],
) -> bool:
    """Verify no original ID appears in multiple splits."""
    split_ids: dict[str, set[tuple[str, str]]] = {}
    for split_name, items in splits.items():
        split_ids[split_name] = {(cls, oid) for cls, oid, _ in items}

    for s1 in split_ids:
        for s2 in split_ids:
            if s1 >= s2:
                continue
            overlap = split_ids[s1] & split_ids[s2]
            if overlap:
                print(f"LEAKAGE: {len(overlap)} IDs shared between {s1} and {s2}")
                print(f"  Examples: {list(overlap)[:5]}")
                return False

    print("No leakage detected. All original IDs are split-exclusive.")
    return True


def build_splits(
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    copy_files: bool = True,
) -> Path:
    """Main entry point. Build splits and write to data/processed/.

    Returns path to the manifest file.
    """
    print(f"Building splits with seed={seed}, ratios={train_ratio}/{val_ratio}/{1-train_ratio-val_ratio:.2f}")

    grouped = collect_images_by_original_id(CLASS_DIRS)
    for cls, ids in grouped.items():
        total_imgs = sum(len(v) for v in ids.values())
        print(f"  {cls}: {len(ids)} unique originals, {total_imgs} total images")

    splits = stratified_id_split(grouped, train_ratio, val_ratio, seed)
    verify_no_leakage(splits)

    output_dir = PROCESSED_DATA_DIR / "classification"
    manifest = write_splits(splits, output_dir, copy_files=copy_files)
    manifest["config"] = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1 - train_ratio - val_ratio, 2),
    }

    manifest_path = output_dir / "split_manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        yaml.safe_dump(manifest, f, default_flow_style=False)

    print(f"\nManifest written to {manifest_path}")
    for split_name, counts in manifest["split_counts"].items():
        print(f"  {split_name}: {counts['total']} images, "
              f"unique IDs: {counts['unique_ids_per_class']}")

    return manifest_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build train/val/test splits by original ID")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--no-copy", action="store_true", help="Skip copying files, just write manifest")
    args = parser.parse_args()

    build_splits(
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        copy_files=not args.no_copy,
    )
