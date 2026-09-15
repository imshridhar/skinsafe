"""SkinSafe AI - Actual Held-Out Model Validation Evaluation Script

Evaluates the trained EfficientNet-B0 checkpoint on all 3,799 held-out validation images
with Test-Time Augmentation (TTA) to generate grounded per-class diagnostic metrics.

Outputs:
  - ml/reports/tables/table_per_class_metrics.csv
  - ml/reports/tables/actual_evaluation_metrics.json
  - ml/reports/figures/fig5_confusion_matrix.png
  - ml/reports/figures/graph8_per_class_metrics.png
"""

import json
import os
import sys
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.src.config import load_yaml_config
from ml.src.model import build_model
from ml.src.preprocessing import ISICDataset, predict_with_tta


def evaluate_actual_checkpoint(
    checkpoint_path: str = "ml/checkpoints/efficientnet_b0/best_model.pt",
    val_csv: str = "dataset/splits/validation.csv",
    config_path: str = "ml/configs/training.yaml",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluation] Device: {device}")

    # 1. Load Trained Checkpoint
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(architecture="efficientnet_b0", num_classes=9, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Evaluation] Successfully loaded checkpoint: {checkpoint_path} (Trained Epochs: {ckpt.get('epoch', 30)})")

    # 2. Load Validation Set (3,799 images)
    val_dataset = ISICDataset(val_csv, is_training=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"[Evaluation] Evaluating on {len(val_dataset)} held-out validation images...")

    all_preds = []
    all_probs = []
    all_targets = []

    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch["image"].to(device)
            labels = batch["label_idx"].to(device)

            probs_tta = predict_with_tta(model, images)
            preds = torch.argmax(probs_tta, dim=1)

            all_probs.extend(probs_tta.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    clinical_names = {
        "MEL": "Melanoma (Malignant)",
        "NV": "Melanocytic Nevus (Benign)",
        "BCC": "Basal Cell Carcinoma (Malignant)",
        "AK": "Actinic Keratosis (Pre-cancer)",
        "BKL": "Benign Keratosis (Benign)",
        "DF": "Dermatofibroma (Benign)",
        "VASC": "Vascular Lesion (Benign)",
        "SCC": "Squamous Cell Carcinoma (Malignant)",
    }
    clinical_risks = {
        "MEL": "Malignant (Critical)",
        "NV": "Benign",
        "BCC": "Malignant",
        "AK": "Pre-cancerous",
        "BKL": "Benign",
        "DF": "Benign (Rare)",
        "VASC": "Benign (Rare)",
        "SCC": "Malignant",
    }

    from typing import Dict, Any, cast

    # Generate Scikit-Learn Metrics
    raw_report = classification_report(
        y_true, y_pred, labels=list(range(8)), target_names=class_names, output_dict=True, zero_division=0
    )
    report_dict: Dict[str, Any] = cast(Dict[str, Any], raw_report)

    raw_top1_acc = np.mean(y_true == y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    one_hot = np.eye(9)[y_true][:, :8]
    macro_auc = roc_auc_score(one_hot, y_prob[:, :8], average="macro", multi_class="ovr")

    # Build Table DataFrame
    rows = []
    for c in class_names:
        stats = report_dict[c]
        rows.append({
            "Lesion Type": f"{clinical_names[c]} ({c})",
            "Original Samples": int(stats["support"]),
            "Effective / Augmented": "—",
            "Precision (%)": round(stats["precision"] * 100, 1),
            "Recall / Sensitivity (%)": round(stats["recall"] * 100, 1),
            "F1-score (%)": round(stats["f1-score"] * 100, 1),
            "Clinical Risk Category": clinical_risks[c]
        })

    # Add Macro Average Row
    rows.append({
        "Lesion Type": "Macro Average (All 8 Classes)",
        "Original Samples": len(y_true),
        "Effective / Augmented": "—",
        "Precision (%)": round(report_dict["macro avg"]["precision"] * 100, 1),
        "Recall / Sensitivity (%)": round(report_dict["macro avg"]["recall"] * 100, 1),
        "F1-score (%)": round(report_dict["macro avg"]["f1-score"] * 100, 1),
        "Clinical Risk Category": "Multi-Class Mean"
    })

    actual_df = pd.DataFrame(rows)
    out_dir = Path("ml/reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    actual_df.to_csv(out_dir / "table_per_class_metrics.csv", index=False)

    # Save summary JSON
    summary_data = {
        "raw_top1_accuracy": round(float(raw_top1_acc) * 100, 2),
        "balanced_accuracy": round(balanced_acc * 100, 2),
        "macro_auc_roc": round(float(macro_auc) * 100, 2),
        "macro_precision": round(float(report_dict["macro avg"]["precision"]) * 100, 2),
        "macro_recall": round(float(report_dict["macro avg"]["recall"]) * 100, 2),
        "macro_f1": round(float(report_dict["macro avg"]["f1-score"]) * 100, 2),
        "total_evaluated_images": len(y_true),
        "per_class": {c: {k: round(v * 100 if k != 'support' else v, 2) for k, v in report_dict[c].items()} for c in class_names}
    }
    with open("ml/reports/tables/actual_evaluation_metrics.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    # 3. Plot Actual Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(8)), normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    sns.heatmap(cm, annot=True, fmt=".1%", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Actual Model Normalized Confusion Matrix (Validation N = 3,799)", pad=15, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Pathological Ground Truth", fontsize=11)
    fig.savefig("ml/reports/figures/fig5_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Plot Actual Per-Class Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    x = np.arange(len(class_names))
    width = 0.26
    prec_vals = [report_dict[c]["precision"] * 100 for c in class_names]
    rec_vals = [report_dict[c]["recall"] * 100 for c in class_names]
    f1_vals = [report_dict[c]["f1-score"] * 100 for c in class_names]

    ax.bar(x - width, prec_vals, width, label="Precision (%)", color="#0EA5E9")
    ax.bar(x, rec_vals, width, label="Recall / Sensitivity (%)", color="#10B981")
    ax.bar(x + width, f1_vals, width, label="F1-Score (%)", color="#A855F7")

    ax.set_title("Actual Per-Class Diagnostic Performance on Validation Split (N = 3,799)", pad=15, fontsize=14, fontweight="bold")
    ax.set_xlabel("Lesion Pathology Class", fontsize=12)
    ax.set_ylabel("Metric Score (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.savefig("ml/reports/figures/graph8_per_class_metrics.png", bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 65)
    print("  ACTUAL MODEL VALIDATION EVALUATION RESULTS")
    print("=" * 65)
    print(f"  Total Validation Images Evaluated: {len(y_true):,}")
    print(f"  Raw Top-1 Accuracy:                 {raw_top1_acc * 100:.2f}%")
    print(f"  Balanced Multi-Class Accuracy:      {balanced_acc * 100:.2f}%")
    print(f"  Macro AUC-ROC Score:                {macro_auc * 100:.2f}%")
    print(f"  Macro Average Precision:            {report_dict['macro avg']['precision'] * 100:.2f}%")
    print(f"  Macro Average Recall:               {report_dict['macro avg']['recall'] * 100:.2f}%")
    print(f"  Macro Average F1-Score:             {report_dict['macro avg']['f1-score'] * 100:.2f}%")
    print("=" * 65)

    return summary_data


if __name__ == "__main__":
    evaluate_actual_checkpoint()
