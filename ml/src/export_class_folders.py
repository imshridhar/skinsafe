"""Utility script to export ISIC 2019 dataset into class-wise folder hierarchy.

Creates:
  dataset/class_folders/train/<CLASS_NAME>/<image_id>.jpg
  dataset/class_folders/validation/<CLASS_NAME>/<image_id>.jpg
  dataset/class_folders/calibration/<CLASS_NAME>/<image_id>.jpg

Supports hardlinks, symlinks, or file copies to avoid unnecessary disk duplication.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from tqdm import tqdm

from ml.src.config import load_class_mapping
from ml.src.logging_utils import setup_logger

logger = setup_logger(name="export_class_folders", log_dir="ml/reports/logs")


def export_split_to_class_folders(
    split_csv_path: str,
    output_base_dir: str,
    split_name: str,
    method: str = "hardlink"
) -> None:
    """Exports images from a split CSV into class-named directories."""
    csv_file = Path(split_csv_path)
    if not csv_file.exists():
        logger.error("Split CSV not found: %s", split_csv_path)
        return

    df = pd.read_csv(csv_file)
    split_dir = Path(output_base_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting %d images for split '%s' into %s using method: %s", len(df), split_name, split_dir, method)

    success_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Exporting {split_name}"):
        src_rel_path = str(row["image_path"])
        src_path = Path(src_rel_path)
        if not src_path.exists():
            # Try resolving from REPO_ROOT
            src_path = REPO_ROOT / src_rel_path

        if not src_path.exists():
            continue

        label = str(row["label"]) if pd.notna(row.get("label")) else "UNK"
        target_dir = split_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)

        target_file = target_dir / f"{row['image_id']}.jpg"

        if target_file.exists():
            success_count += 1
            continue

        try:
            if method == "hardlink":
                os.link(src_path, target_file)
            elif method == "symlink":
                os.symlink(src_path.resolve(), target_file)
            else:
                shutil.copy2(src_path, target_file)
            success_count += 1
        except Exception:
            # Fallback to copy if hardlink/symlink is not permitted
            try:
                shutil.copy2(src_path, target_file)
                success_count += 1
            except Exception as e:
                logger.warning("Failed to link/copy %s: %s", src_path, e)

    logger.info("Successfully organized %d / %d images into %s", success_count, len(df), split_dir)


def main():
    parser = argparse.ArgumentParser(description="Export ISIC 2019 dataset to class-wise folder hierarchy")
    parser.add_argument("--output-dir", type=str, default="dataset/class_folders", help="Target folder for class-wise splits")
    parser.add_argument("--method", choices=["hardlink", "symlink", "copy"], default="hardlink", help="Link/copy method (hardlink saves disk space)")
    args = parser.parse_args()

    splits = [
        ("dataset/splits/train.csv", "train"),
        ("dataset/splits/validation.csv", "validation"),
        ("dataset/splits/calibration.csv", "calibration")
    ]

    for csv_path, split_name in splits:
        export_split_to_class_folders(csv_path, args.output_dir, split_name, method=args.method)

    print("\n" + "=" * 60)
    print(f"Class-wise folders created at: {Path(args.output_dir).resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
