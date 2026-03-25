"""
merge_datasets.py

Merges multiple per-video YOLO datasets (each in data/dataset_N/) into a single
combined dataset ready for model training, with optional train/val split.

Expected input structure per dataset:
    data/
        dataset_1/
            images/         # frame_XXXXXX.png
            labels/         # frame_XXXXXX.txt  (YOLO format)
            data.yaml
        dataset_2/
            ...

Output structure:
    merged/
        images/
            train/
            val/
        labels/
            train/
            val/
        data.yaml
"""

import os
import random
import shutil
from pathlib import Path

import yaml

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_ROOT = Path("data")  # folder containing dataset_1, dataset_2, ...
OUTPUT_DIR = Path("merged")  # where the merged dataset will be written
VAL_SPLIT = 0.15  # fraction of images reserved for validation
SEED = 42  # reproducibility

# ── Helpers ───────────────────────────────────────────────────────────────────


def load_classes(yaml_path: Path) -> list[str]:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {})
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return names  # already a list


def collect_datasets(data_root: Path) -> list[Path]:
    """Return sorted list of dataset_N subdirectories."""
    return sorted(
        p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("dataset_")
    )


def merge(data_root: Path, output_dir: Path, val_split: float, seed: int):
    random.seed(seed)

    datasets = collect_datasets(data_root)
    if not datasets:
        raise FileNotFoundError(f"No dataset_* folders found under '{data_root}'")

    print(f"Found {len(datasets)} dataset(s): {[d.name for d in datasets]}")

    # Read class names from first dataset (assumed consistent across all)
    class_names = load_classes(datasets[0] / "data.yaml")
    print(f"Classes: {class_names}")

    # Create output directories
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_count = val_count = 0

    for ds in datasets:
        images_dir = ds / "images"
        labels_dir = ds / "labels"

        image_files = sorted(images_dir.glob("*.png")) + sorted(
            images_dir.glob("*.jpg")
        )

        if not image_files:
            print(f"  [{ds.name}] No images found, skipping.")
            continue

        # Shuffle and split
        random.shuffle(image_files)
        n_val = max(1, int(len(image_files) * val_split))
        val_set = set(f.name for f in image_files[:n_val])

        for img_path in image_files:
            split = "val" if img_path.name in val_set else "train"

            # Prefix filename with dataset name to avoid collisions
            new_stem = f"{ds.name}_{img_path.stem}"
            new_img_name = new_stem + img_path.suffix
            new_lbl_name = new_stem + ".txt"

            # Copy image
            shutil.copy2(img_path, output_dir / "images" / split / new_img_name)

            # Copy label (if it exists)
            lbl_src = labels_dir / (img_path.stem + ".txt")
            lbl_dst = output_dir / "labels" / split / new_lbl_name
            if lbl_src.exists():
                shutil.copy2(lbl_src, lbl_dst)
            else:
                # Create empty label file (background image)
                lbl_dst.touch()

            if split == "train":
                train_count += 1
            else:
                val_count += 1

        print(
            f"  [{ds.name}] {len(image_files)} images → "
            f"{len(image_files) - n_val} train / {n_val} val"
        )

    # Write merged data.yaml
    yaml_out = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(yaml_out, f, default_flow_style=False, sort_keys=False)

    print(f"\nDone! Merged dataset written to '{output_dir}/'")
    print(f"  Total train: {train_count} | val: {val_count}")
    print(f"  data.yaml:   {output_dir / 'data.yaml'}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    merge(DATA_ROOT, OUTPUT_DIR, VAL_SPLIT, SEED)
