import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Ensure repository root is in sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ml.src.config import load_class_mapping
from ml.src.logging_utils import setup_logger, write_json_report
from ml.src.seed import set_seed



logger = setup_logger(name="create_splits", log_dir="ml/reports/logs")


class DisjointSetUnion:
    """Disjoint Set Union (DSU) to compute connected components across lesion_ids and SHA-256 hashes."""
    def __init__(self):
        self.parent = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: str, item2: str) -> None:
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            self.parent[root1] = root2


def build_leakage_proof_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Clusters images into disjoint components so that no SHA-256 duplicate or lesion_id crosses splits."""
    df = df.copy()
    dsu = DisjointSetUnion()
    
    # 1. Connect images by SHA-256 and by lesion_id
    for _, row in df.iterrows():
        img_node = f"img_{row['image_id']}"
        hash_node = f"hash_{row['sha256']}"
        dsu.union(img_node, hash_node)
        
        lesion = row.get("lesion_id")
        if pd.notna(lesion) and str(lesion).strip() != "":
            lesion_node = f"lesion_{str(lesion).strip()}"
            dsu.union(img_node, lesion_node)
            
    # 2. Assign canonical cluster root as group_id
    df["group_id"] = [dsu.find(f"img_{img_id}") for img_id in df["image_id"]]
    return df


def verify_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, calib_df: pd.DataFrame) -> Dict[str, Any]:
    """Strictly checks for 0% data leakage across all splits."""
    train_groups = set(train_df["group_id"].unique())
    val_groups = set(val_df["group_id"].unique())
    calib_groups = set(calib_df["group_id"].unique())
    
    train_hashes = set(train_df["sha256"].unique())
    val_hashes = set(val_df["sha256"].unique())
    calib_hashes = set(calib_df["sha256"].unique())
    
    # Lesion IDs (ignoring nulls)
    get_lesions = lambda df: set(df["lesion_id"].dropna().astype(str).unique()) - {""}
    train_lesions = get_lesions(train_df)
    val_lesions = get_lesions(val_df)
    calib_lesions = get_lesions(calib_df)
    
    leakage_results = {
        "group_crossover_train_val": len(train_groups & val_groups),
        "group_crossover_train_calib": len(train_groups & calib_groups),
        "group_crossover_val_calib": len(val_groups & calib_groups),
        "hash_crossover_train_val": len(train_hashes & val_hashes),
        "hash_crossover_train_calib": len(train_hashes & calib_hashes),
        "hash_crossover_val_calib": len(val_hashes & calib_hashes),
        "lesion_crossover_train_val": len(train_lesions & val_lesions),
        "lesion_crossover_train_calib": len(train_lesions & calib_lesions),
        "lesion_crossover_val_calib": len(val_lesions & calib_lesions),
        "is_leakage_free": True
    }
    
    total_leaks = sum(v for k, v in leakage_results.items() if k != "is_leakage_free")
    if total_leaks > 0:
        leakage_results["is_leakage_free"] = False
        raise ValueError(f"FATAL: Data leakage detected across splits! Audit details: {leakage_results}")
        
    return leakage_results


def create_splits(
    manifest_path: str = "ml/reports/dataset_manifest.csv",
    output_dir: str = "dataset/splits",
    seed: int = 42
) -> Dict[str, Any]:
    """Generates leakage-free Stratified Group splits (Train: 70%, Val: 15%, Calib: 15%)."""
    set_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading manifest from %s...", manifest_path)
    manifest_df = pd.read_csv(manifest_path)
    
    # Filter valid labeled records
    valid_df = manifest_df[manifest_df["is_valid"] & manifest_df["label"].notna()].copy()
    logger.info("Total valid labeled samples for splitting: %d", len(valid_df))
    
    # Build DSU graph connected components
    valid_df = build_leakage_proof_groups(valid_df)
    unique_groups = valid_df["group_id"].nunique()
    logger.info("Total unique DSU connected component groups: %d", unique_groups)
    
    # Use StratifiedGroupKFold with 20 folds to construct a 70% / 15% / 15% split
    # (14 folds = 70% Train, 3 folds = 15% Val, 3 folds = 15% Calib)
    n_splits = 20
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    X = np.zeros(len(valid_df))
    y = valid_df["class_index"].values
    groups = valid_df["group_id"].values
    
    fold_indices = np.zeros(len(valid_df), dtype=int)
    for fold, (_, val_idx) in enumerate(sgkf.split(X, y, groups)):
        fold_indices[val_idx] = fold
        
    valid_df["fold"] = fold_indices
    
    # Assign folds to splits
    train_mask = valid_df["fold"] < 14
    val_mask = (valid_df["fold"] >= 14) & (valid_df["fold"] < 17)
    calib_mask = valid_df["fold"] >= 17
    
    train_df = valid_df[train_mask].copy()
    val_df = valid_df[val_mask].copy()
    calib_df = valid_df[calib_mask].copy()
    
    # Verify zero leakage
    logger.info("Verifying 0% data leakage across train, val, and calib...")
    leakage_audit = verify_no_leakage(train_df, val_df, calib_df)
    logger.info("LEAKAGE VERIFICATION PASSED: All cross-split intersections are exactly 0.")
    
    # Export split CSVs
    train_path = out_dir / "train.csv"
    val_path = out_dir / "validation.csv"
    calib_path = out_dir / "calibration.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    calib_df.to_csv(calib_path, index=False)
    
    logger.info("Saved train split (%d samples) to %s", len(train_df), train_path)
    logger.info("Saved val split (%d samples) to %s", len(val_df), val_path)
    logger.info("Saved calib split (%d samples) to %s", len(calib_df), calib_path)
    
    # Generate split report
    split_summary = {
        "random_seed": seed,
        "strategy": "StratifiedGroupKFold with Disjoint Set Union Clustering (14 Train, 3 Val, 3 Calib)",
        "total_samples": len(valid_df),
        "train_samples": len(train_df),
        "train_percentage": round(len(train_df) / len(valid_df) * 100, 2),
        "val_samples": len(val_df),
        "val_percentage": round(len(val_df) / len(valid_df) * 100, 2),
        "calib_samples": len(calib_df),
        "calib_percentage": round(len(calib_df) / len(valid_df) * 100, 2),
        "class_distribution": {
            "train": train_df["label"].value_counts().to_dict(),
            "val": val_df["label"].value_counts().to_dict(),
            "calib": calib_df["label"].value_counts().to_dict()
        },
        "leakage_audit": leakage_audit
    }
    
    write_json_report(split_summary, str(out_dir / "split_report.json"))
    logger.info("Split report saved to: %s", out_dir / "split_report.json")
    return split_summary


if __name__ == "__main__":
    summary = create_splits()
    print("\n" + "=" * 60)
    print("LEAKAGE-FREE SPLIT GENERATION SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
