import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, cast

# Ensure repository root is in sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.src.config import load_class_mapping, load_yaml_config
from ml.src.logging_utils import setup_logger, write_json_report
from ml.src.losses import FocalLoss, compute_class_weights
from ml.src.model import build_model, EfficientNetB0Classifier
from ml.src.preprocessing import ISICDataset, get_base_transform, get_training_transform, predict_with_tta
from ml.src.seed import seed_worker, set_seed

logger = setup_logger(name="train_engine", log_dir="ml/reports/logs")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 9,
    use_tta: bool = True
) -> Dict[str, float]:
    """Evaluates classifier on validation dataloader, with optional Test-Time Augmentation (TTA).

    Args:
        model: PyTorch classifier module.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Active compute device.
        num_classes: Total class count (9).
        use_tta: If True, uses 4-view test-time augmentation.

    Returns:
        Dictionary of validation metrics.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label_idx"].to(device, non_blocking=True)

            if use_tta:
                probs_tensor = predict_with_tta(model, images)
                logits = torch.log(probs_tensor + 1e-7)
                loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                probs_tensor = torch.softmax(logits, dim=1)

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(probs_tensor, dim=1).cpu().numpy()
            probs = probs_tensor.cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    all_targets_arr = np.array(all_targets)
    all_preds_arr = np.array(all_preds)
    all_probs_arr = np.array(all_probs)

    avg_loss = total_loss / len(all_targets_arr)
    top1_acc = float(np.mean(all_preds_arr == all_targets_arr))
    balanced_acc = balanced_accuracy_score(all_targets_arr, all_preds_arr)

    # Compute Macro ROC-AUC across present classes
    present_classes = np.unique(all_targets_arr)
    try:
        if len(present_classes) > 1:
            one_hot_targets = np.eye(num_classes)[all_targets_arr]
            macro_auc = float(roc_auc_score(one_hot_targets[:, present_classes], all_probs_arr[:, present_classes], average="macro", multi_class="ovr"))
        else:
            macro_auc = 0.5
    except Exception:
        macro_auc = 0.5

    # Melanoma Sensitivity (Class 0: MEL)
    mel_mask = (all_targets_arr == 0)
    mel_recall = float(np.mean(all_preds_arr[mel_mask] == 0)) if np.sum(mel_mask) > 0 else 0.0

    return {
        "loss": avg_loss,
        "top1_accuracy": top1_acc,
        "balanced_accuracy": balanced_acc,
        "macro_auc": macro_auc,
        "melanoma_sensitivity": mel_recall
    }


def train(config_path: str = "ml/configs/training.yaml", dry_run: bool = False) -> Dict[str, Any]:
    """Runs end-to-end 2-Stage training pipeline with gradient accumulation, warmup, and early stopping.

    Args:
        config_path: Path to YAML training configuration.
        dry_run: If True, executes 1 epoch on 1 batch for pipeline validation without touching production checkpoint.

    Returns:
        Dictionary containing best model metric and training epoch records.
    """
    raw_cfg = load_yaml_config(config_path)
    # Support both nested 'training:' and flat config structures seamlessly
    cfg = raw_cfg.get("training", raw_cfg)

    output_dir_str = cfg.get("output_dir", "ml/checkpoints/efficientnet_b0")
    arch_name = cfg.get("model_architecture", "efficientnet_b0")
    output_dir = Path(output_dir_str)
    if not output_dir_str.endswith(arch_name):
        output_dir = output_dir / arch_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds for reproducibility
    seed_val = int(cfg.get("seed", 42))
    g = set_seed(seed_val)

    # Hardware configuration
    hw_cfg = cfg.get("hardware", {})
    device_str = hw_cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info("Training on device: %s (%s)", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")

    # Load split CSV paths
    splits_cfg = cfg.get("splits", {})
    dataset_cfg = cfg.get("dataset", {})
    train_csv = splits_cfg.get("train_path", dataset_cfg.get("train_split", "dataset/splits/train.csv"))
    val_csv = splits_cfg.get("val_path", dataset_cfg.get("val_split", "dataset/splits/validation.csv"))

    # In dry-run mode, create tiny subsets
    if dry_run:
        logger.info("--- DRY-RUN MODE ACTIVATED: Subsetting to 32 samples for quick smoke test ---")
        train_df = pd.read_csv(train_csv).head(32)
        val_df = pd.read_csv(val_csv).head(16)
        dry_train_csv = output_dir / "dry_train.csv"
        dry_val_csv = output_dir / "dry_val.csv"
        train_df.to_csv(dry_train_csv, index=False)
        val_df.to_csv(dry_val_csv, index=False)
        train_csv = str(dry_train_csv)
        val_csv = str(dry_val_csv)

    train_dataset = ISICDataset(train_csv, is_training=True)
    val_dataset = ISICDataset(val_csv, is_training=False)

    stages_cfg = cfg.get("stages", {})
    stage_a_cfg = stages_cfg.get("stage_a", {})
    stage_b_cfg = stages_cfg.get("stage_b", {})

    batch_size = hw_cfg.get("batch_size", stage_a_cfg.get("batch_size", 8))
    accum_steps = hw_cfg.get("gradient_accumulation_steps", stage_a_cfg.get("gradient_accumulation_steps", 2))
    num_workers = 0 if sys.platform == "win32" else hw_cfg.get("num_workers", 0)
    pin_mem = hw_cfg.get("pin_memory", True)
    mixed_prec = hw_cfg.get("mixed_precision", True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem
    )

    logger.info("Loaded Train samples: %d, Validation samples: %d", len(train_dataset), len(val_dataset))
    logger.info("Batch Size: %d, Gradient Accumulation Steps: %d (Effective Batch Size: %d)", batch_size, accum_steps, batch_size * accum_steps)

    # Compute class weights for Focal Loss across all num_classes (0..num_classes-1)
    train_df = pd.read_csv(train_csv)
    num_classes = int(cfg.get("num_classes", 9))
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    for idx, count in train_df["class_index"].value_counts().items():
        idx_int = int(str(idx))
        if 0 <= idx_int < num_classes:
            class_counts[idx_int] = float(count)
    class_counts = torch.where(class_counts == 0, torch.ones_like(class_counts), class_counts)
    class_weights = compute_class_weights(class_counts).to(device)

    loss_cfg = cfg.get("loss", {})
    criterion = FocalLoss(
        gamma=float(loss_cfg.get("gamma", 2.0)),
        alpha=class_weights if loss_cfg.get("use_class_weights", True) else None,
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.05))
    )

    # Build Model
    built_model = build_model(
        architecture=arch_name,
        num_classes=num_classes,
        pretrained=cfg.get("pretrained", True)
    ).to(device)
    model: EfficientNetB0Classifier = cast(EfficientNetB0Classifier, built_model)

    scaler = torch.amp.GradScaler("cuda", enabled=(mixed_prec and device.type == "cuda"))

    # Optimizer with differentiated learning rates
    opt_cfg = cfg.get("optimizer", {})
    lr_head = float(opt_cfg.get("lr_head", 0.001))
    lr_backbone = float(opt_cfg.get("lr_backbone", 0.0001))
    weight_decay = float(opt_cfg.get("weight_decay", 0.01))

    optimizer = torch.optim.AdamW([
        {"params": model.classifier.parameters(), "lr": lr_head},
        {"params": model.features.parameters(), "lr": lr_backbone}
    ], weight_decay=weight_decay)

    stage_a_epochs = int(stage_a_cfg.get("epochs", 5))
    stage_b_epochs = int(stage_b_cfg.get("epochs", 25))
    total_epochs = 1 if dry_run else (stage_a_epochs + stage_b_epochs)
    
    sched_cfg = cfg.get("scheduler", {})
    eta_min = float(sched_cfg.get("eta_min", 1e-6))
    warmup_epochs = int(sched_cfg.get("warmup_epochs", min(3, max(1, total_epochs // 5)))) if not dry_run else 1

    if not dry_run and total_epochs > warmup_epochs:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)

    # Early stopping config
    es_cfg = cfg.get("early_stopping", {})
    patience = int(es_cfg.get("patience", 7))
    min_delta = float(es_cfg.get("min_delta", 0.001))
    epochs_no_improve = 0

    best_macro_auc = 0.0
    history = []

    # Stage A: Warmup
    logger.info("--- Starting Stage A: Warmup (Head Training Only) ---")
    model.freeze_backbone()

    for epoch in range(1, total_epochs + 1):
        if not dry_run and epoch == stage_a_epochs + 1:
            unfreeze_blocks = int(stage_b_cfg.get("unfreeze_blocks", 4))
            logger.info("--- Starting Stage B: Fine-Tuning (Unfreezing Top %d Blocks) ---", unfreeze_blocks)
            model.unfreeze_top_blocks(unfreeze_blocks)

        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}")):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label_idx"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(mixed_prec and device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)
                scaled_batch_loss = loss / accum_steps

            scaled_loss = cast(torch.Tensor, scaler.scale(scaled_batch_loss))
            scaled_loss.backward()
            epoch_loss += loss.item() * len(labels)

            # Gradient Accumulation Step
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scheduler.step()
        train_loss = epoch_loss / len(train_dataset)
        val_metrics = evaluate_model(model, val_loader, criterion, device, num_classes, use_tta=True)
        elapsed = time.time() - start_time

        logger.info(
            "Epoch %d/%d (%.1fs) - Train Loss: %.4f | Val Loss: %.4f | Val Balanced Acc: %.4f | Val Macro AUC (TTA): %.4f | Mel Sensitivity: %.4f",
            epoch, total_epochs, elapsed, train_loss, val_metrics["loss"], val_metrics["balanced_accuracy"], val_metrics["macro_auc"], val_metrics["melanoma_sensitivity"]
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_top1_accuracy": val_metrics["top1_accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_auc": val_metrics["macro_auc"],
            "val_melanoma_sensitivity": val_metrics["melanoma_sensitivity"],
            "elapsed_seconds": elapsed
        }
        history.append(record)

        # Checkpoint Best Model & Early Stopping
        current_auc = val_metrics["macro_auc"]
        if current_auc >= (best_macro_auc + min_delta):
            best_macro_auc = current_auc
            epochs_no_improve = 0
            ckpt_name = "dry_run_model.pt" if dry_run else "best_model.pt"
            checkpoint_path = output_dir / ckpt_name
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_auc": best_macro_auc,
                "config": cfg
            }, checkpoint_path)
            logger.info("Saved new BEST checkpoint to %s (Val Macro AUC: %.4f)", checkpoint_path, best_macro_auc)
        else:
            epochs_no_improve += 1
            if not dry_run and epochs_no_improve >= patience and epoch > stage_a_epochs:
                logger.info("Early stopping triggered after %d epochs without improvement (Best Macro AUC: %.4f).", epochs_no_improve, best_macro_auc)
                break

    # Save training history report
    if not dry_run:
        history_path = output_dir / "training_history.json"
        write_json_report({"history": history, "best_macro_auc": best_macro_auc}, str(history_path))
        logger.info("Training complete. Best Macro AUC: %.4f", best_macro_auc)

        # Automatically calibrate model and recompute performance table for full training runs
        try:
            from ml.src.calibrate_model import calibrate_trained_model
            from ml.src.evaluate_actual_metrics import evaluate_actual_checkpoint
            from ml.src.show_metrics import display_terminal_metrics

            logger.info("--- Automatically Running Post-Training Calibration ---")
            calibrate_trained_model(checkpoint_path=str(output_dir / "best_model.pt"))

            logger.info("--- Automatically Generating Dynamic Per-Class Metrics Table ---")
            evaluate_actual_checkpoint(checkpoint_path=str(output_dir / "best_model.pt"))

            logger.info("Performance table and calibration parameters dynamically refreshed for newly trained weights!")
            display_terminal_metrics()

        except Exception as e:
            logger.warning("Post-training evaluation note: %s", e)
    else:
        logger.info("--- Dry-run complete. Saved to %s. Production best_model.pt preserved! ---", output_dir / "dry_run_model.pt")

    return {"best_macro_auc": best_macro_auc, "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISIC 2019 Diagnostic Classifier Training Engine")
    parser.add_argument("--config", type=str, default="ml/configs/training.yaml", help="Path to YAML configuration")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch on a tiny subset to verify pipeline health without overwriting best_model.pt")
    args = parser.parse_args()
    train(config_path=args.config, dry_run=args.dry_run)
