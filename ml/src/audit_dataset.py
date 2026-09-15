import os
import sys
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure repository root is in sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ml.src.config import load_class_mapping, get_inv_class_mapping
from ml.src.logging_utils import setup_logger, write_json_report



logger = setup_logger(name="audit_dataset", log_dir="ml/reports/logs")


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_image(filepath: Path) -> Dict[str, Any]:
    """Inspects a single image file for validity, dimensions, and SHA-256."""
    result: Dict[str, Any] = {
        "is_valid": True,
        "validation_error": None,
        "width": 0,
        "height": 0,
        "mode": "",
        "file_size_bytes": 0,
        "sha256": ""
    }
    
    if not filepath.exists():
        result["is_valid"] = False
        result["validation_error"] = "File not found on disk"
        return result

    try:
        result["file_size_bytes"] = filepath.stat().st_size
        result["sha256"] = compute_sha256(filepath)
        
        # Pillow verification
        with Image.open(filepath) as img:
            img.verify()
            
        # Re-open to read dimensions & mode after verify()
        with Image.open(filepath) as img:
            result["width"], result["height"] = img.size
            result["mode"] = img.mode
            
            if img.mode not in ("RGB", "L", "RGBA"):
                result["is_valid"] = False
                result["validation_error"] = f"Unsupported image mode: {img.mode}"
                
    except Exception as e:
        result["is_valid"] = False
        result["validation_error"] = f"Corrupted image decode error: {str(e)}"
        
    return result


def parse_ground_truth(gt_path: Path, class_mapping: Dict[int, str]) -> pd.DataFrame:
    """Parses ground-truth CSV and extracts canonical class labels."""
    df = pd.read_csv(gt_path)
    df["image"] = df["image"].astype(str).str.strip()
    
    class_cols = [class_mapping[i] for i in range(len(class_mapping)) if class_mapping[i] in df.columns]
    
    records = []
    for _, row in df.iterrows():
        img_id = row["image"]
        active_classes = [c for c in class_cols if float(row[c]) == 1.0]
        
        if len(active_classes) == 1:
            label = active_classes[0]
            label_error = None
        elif len(active_classes) == 0:
            label = None
            label_error = "No active class in ground truth (sum=0)"
        else:
            label = active_classes[0]  # Record first but flag conflict
            label_error = f"Multiple active classes: {active_classes}"
            
        records.append({
            "image": img_id,
            "label": label,
            "label_error": label_error
        })
        
    return pd.DataFrame(records)


def run_audit(
    dataset_dir: str = "dataset",
    output_dir: str = "ml/reports",
    max_workers: int = 16
) -> Dict[str, Any]:
    """Runs complete dataset audit on ISIC 2019 train and test sets.

    Args:
        dataset_dir: Root dataset folder.
        output_dir: Output directory for reports.
        max_workers: ThreadPool workers for parallel image inspection.

    Returns:
        Summary audit dictionary.
    """
    data_path = Path(dataset_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    class_mapping = load_class_mapping()
    inv_class_map = get_inv_class_mapping()
    
    train_img_dir = data_path / "raw" / "train" / "images"
    test_img_dir = data_path / "raw" / "test" / "images"
    train_gt_path = data_path / "labels" / "ISIC_2019_Training_GroundTruth.csv"
    train_meta_path = data_path / "metadata" / "ISIC_2019_Training_Metadata.csv"
    test_meta_path = data_path / "metadata" / "ISIC_2019_Test_Metadata.csv"
    
    logger.info("Starting ISIC 2019 dataset audit...")
    logger.info("Training images path: %s", train_img_dir)
    logger.info("Test images path (Locked): %s", test_img_dir)
    
    # 1. Parse ground truth & metadata
    logger.info("Parsing ground truth and metadata CSVs...")
    train_gt_df = parse_ground_truth(train_gt_path, class_mapping)
    train_meta_df = pd.read_csv(train_meta_path)
    train_meta_df["image"] = train_meta_df["image"].astype(str).str.strip()
    
    train_merged = pd.merge(train_gt_df, train_meta_df, on="image", how="outer")
    logger.info("Total training records in CSVs: %d", len(train_merged))
    
    # 2. Parallel Image Inspection for Training Images
    logger.info("Inspecting %d training images with %d workers...", len(train_merged), max_workers)
    
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, row in train_merged.iterrows():
            img_id = row["image"]
            img_path = train_img_dir / f"{img_id}.jpg"
            tasks.append((img_id, img_path, executor.submit(inspect_image, img_path)))
            
        manifest_rows = []
        for img_id, img_path, future in tqdm(tasks, desc="Auditing Training Images"):
            inspection = future.result()
            row_meta = train_merged[train_merged["image"] == img_id].iloc[0]
            
            label = row_meta.get("label")
            class_idx = inv_class_map.get(label) if label in inv_class_map else None
            
            val_err = inspection["validation_error"] or row_meta.get("label_error")
            is_valid = inspection["is_valid"] and (row_meta.get("label_error") is None)
            
            manifest_rows.append({
                "image_id": img_id,
                "image_path": str(img_path.relative_to(Path("."))),
                "split_source": "train",
                "label": label,
                "class_index": class_idx,
                "age_approx": row_meta.get("age_approx"),
                "anatom_site_general": row_meta.get("anatom_site_general"),
                "sex": row_meta.get("sex"),
                "lesion_id": row_meta.get("lesion_id"),
                "sha256": inspection["sha256"],
                "width": inspection["width"],
                "height": inspection["height"],
                "mode": inspection["mode"],
                "file_size_bytes": inspection["file_size_bytes"],
                "is_valid": is_valid,
                "validation_error": val_err,
                "preprocessing_version": "v1.0"
            })
            
    manifest_df = pd.DataFrame(manifest_rows)
    
    # 3. Duplicate Detection via SHA-256
    logger.info("Analyzing SHA-256 hashes for duplicate images...")
    hash_counts = manifest_df["sha256"].value_counts()
    duplicate_hashes = hash_counts[hash_counts > 1].index.tolist()
    
    manifest_df["is_duplicate"] = manifest_df["sha256"].isin(duplicate_hashes)
    manifest_df["duplicate_group"] = manifest_df["sha256"].where(manifest_df["is_duplicate"], None)
    
    # Export Manifest
    manifest_csv = out_path / "dataset_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    logger.info("Saved dataset manifest to: %s", manifest_csv)
    
    # 4. Audit Test Set (Locked Evaluation Data)
    logger.info("Auditing locked official test set...")
    test_meta_df = pd.read_csv(test_meta_path)
    test_meta_df["image"] = test_meta_df["image"].astype(str).str.strip()
    
    test_tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, row in test_meta_df.iterrows():
            img_id = row["image"]
            img_path = test_img_dir / f"{img_id}.jpg"
            test_tasks.append((img_id, img_path, executor.submit(inspect_image, img_path)))
            
        test_manifest_rows = []
        for img_id, img_path, future in tqdm(test_tasks, desc="Auditing Test Images"):
            inspection = future.result()
            row_meta = test_meta_df[test_meta_df["image"] == img_id].iloc[0]
            
            test_manifest_rows.append({
                "image_id": img_id,
                "image_path": str(img_path.relative_to(Path("."))),
                "split_source": "official_test",
                "label": None,  # Locked
                "class_index": None,
                "age_approx": row_meta.get("age_approx"),
                "anatom_site_general": row_meta.get("anatom_site_general"),
                "sex": row_meta.get("sex"),
                "lesion_id": row_meta.get("lesion_id"),
                "sha256": inspection["sha256"],
                "width": inspection["width"],
                "height": inspection["height"],
                "mode": inspection["mode"],
                "file_size_bytes": inspection["file_size_bytes"],
                "is_valid": inspection["is_valid"],
                "validation_error": inspection["validation_error"],
                "locked": True,
                "preprocessing_version": "v1.0"
            })
            
    test_manifest_df = pd.DataFrame(test_manifest_rows)
    test_manifest_csv = out_path / "test_manifest.csv"
    test_manifest_df.to_csv(test_manifest_csv, index=False)
    logger.info("Saved locked test manifest to: %s", test_manifest_csv)
    
    # 5. Summary Statistics & Reports
    class_dist = manifest_df["label"].value_counts().to_dict()
    class_dist_df = manifest_df["label"].value_counts().reset_index()
    class_dist_df.columns = ["label", "count"]
    class_dist_df.to_csv(out_path / "class_distribution.csv", index=False)
    
    corrupt_df = manifest_df[~manifest_df["is_valid"]]
    corrupt_df.to_csv(out_path / "corrupt_images.csv", index=False)
    
    dup_df = manifest_df[manifest_df["is_duplicate"]].sort_values("sha256")
    dup_df.to_csv(out_path / "duplicate_groups.csv", index=False)
    
    audit_summary = {
        "dataset": "ISIC 2019",
        "total_train_images": len(manifest_df),
        "valid_train_images": int(manifest_df["is_valid"].sum()),
        "corrupt_train_images": int((~manifest_df["is_valid"]).sum()),
        "total_test_images_locked": len(test_manifest_df),
        "valid_test_images": int(test_manifest_df["is_valid"].sum()),
        "class_distribution": class_dist,
        "unique_sha256_hashes": int(manifest_df["sha256"].nunique()),
        "duplicate_images_count": int(manifest_df["is_duplicate"].sum()),
        "duplicate_groups_count": len(duplicate_hashes),
        "missing_metadata": {
            "age_approx": int(manifest_df["age_approx"].isna().sum()),
            "anatom_site_general": int(manifest_df["anatom_site_general"].isna().sum()),
            "sex": int(manifest_df["sex"].isna().sum()),
            "lesion_id": int(manifest_df["lesion_id"].isna().sum())
        },
        "resolution_stats": {
            "min_width": int(manifest_df["width"].min()),
            "max_width": int(manifest_df["width"].max()),
            "min_height": int(manifest_df["height"].min()),
            "max_height": int(manifest_df["height"].max())
        }
    }
    
    write_json_report(audit_summary, str(out_path / "dataset_audit.json"))
    logger.info("Dataset audit completed successfully! Summary written to: %s", out_path / "dataset_audit.json")
    return audit_summary


if __name__ == "__main__":
    summary = run_audit()
    print("\n" + "=" * 60)
    print("ISIC 2019 DATASET AUDIT SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
