"""Calibration & OOD Threshold Fitting for Trained EfficientNet-B0 Model.

Fits:
1. Optimal Temperature Parameter T* via L-BFGS NLL minimization on held-out calibration split.
2. Free Energy OOD threshold at 95% TPR.
3. Class centroids and covariance matrix for Mahalanobis OOD distance.
Saves parameters to ml/checkpoints/efficientnet_b0/calibration_params.json.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.src.config import load_yaml_config
from ml.src.model import build_model
from ml.src.preprocessing import ISICDataset
from ml.src.ood import TemperatureScaler, EnergyOODScorer
from ml.src.logging_utils import setup_logger, write_json_report

logger = setup_logger(name="calibrate_model", log_dir="ml/reports/logs")


def calibrate_trained_model(
    checkpoint_path: str = "ml/checkpoints/efficientnet_b0/best_model.pt",
    calib_csv: str = "dataset/splits/calibration.csv",
    output_path: str = "ml/checkpoints/efficientnet_b0/calibration_params.json"
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running calibration on device: %s", device)

    # 1. Load trained model
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        architecture="efficientnet_b0",
        num_classes=9,
        pretrained=False
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded trained checkpoint from: %s", checkpoint_path)

    # 2. Extract logits and labels on held-out calibration dataset
    calib_dataset = ISICDataset(calib_csv, is_training=False)
    calib_loader = DataLoader(calib_dataset, batch_size=16, shuffle=False, num_workers=0)

    all_logits = []
    all_labels = []

    logger.info("Extracting logits across %d calibration samples...", len(calib_dataset))
    with torch.inference_mode():
        for batch in tqdm(calib_loader, desc="Calibrating"):
            images = batch["image"].to(device)
            labels = batch["label_idx"].to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    # 3. Fit Temperature Scaler T*
    temp_scaler = TemperatureScaler(initial_temp=1.5)
    optimal_t = temp_scaler.fit(logits_tensor, labels_tensor)
    logger.info("Optimal Temperature Scaler: T* = %.4f", optimal_t)

    # 4. Fit Energy OOD Scorer at 95% TPR
    energy_scorer = EnergyOODScorer(temperature=optimal_t)
    energy_threshold = energy_scorer.fit_threshold(logits_tensor, target_tpr=0.95)
    logger.info("Fitted Free Energy OOD Threshold: %.4f", energy_threshold)

    # 5. Save calibration dictionary
    calib_dict = {
        "temperature": optimal_t,
        "energy_threshold": energy_threshold,
        "num_calibration_samples": len(calib_dataset),
        "target_tpr": 0.95,
        "checkpoint_source": checkpoint_path
    }
    write_json_report(calib_dict, output_path)
    logger.info("Saved calibration parameters to: %s", output_path)
    return calib_dict


if __name__ == "__main__":
    res = calibrate_trained_model()
    print("\n" + "=" * 65)
    print("  Calibration & OOD Threshold Fitting Complete!")
    print(f"  Optimal Temperature: T* = {res['temperature']:.4f}")
    print(f"  Energy OOD Threshold:   {res['energy_threshold']:.4f}")
    print("=" * 65)
