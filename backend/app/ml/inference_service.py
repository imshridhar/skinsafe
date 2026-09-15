import time
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from ml.src.config import load_class_mapping
from ml.src.gradcam import GradCAM
from ml.src.model import EfficientNetB0Classifier
from ml.src.ood import EnergyOODScorer, TemperatureScaler
from ml.src.preprocessing import preprocess_image_bytes
from backend.app.ml.model_loader import ModelManager
from backend.app.utils.security import validate_image_magic_bytes


class InferenceService:
    """Production Inference Service coordinating preprocessing, prediction, OOD, and Grad-CAM."""
    def __init__(self, upload_dir: Path, gradcam_dir: Path):
        self.upload_dir = Path(upload_dir)
        self.gradcam_dir = Path(gradcam_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.gradcam_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_mapping = load_class_mapping()
        self.class_names = {
            "MEL": "Melanoma (Malignant)",
            "NV": "Melanocytic Nevus (Benign)",
            "BCC": "Basal Cell Carcinoma (Malignant)",
            "AK": "Actinic Keratosis (Pre-cancerous)",
            "BKL": "Benign Keratosis",
            "DF": "Dermatofibroma (Benign)",
            "VASC": "Vascular Lesion (Benign)",
            "SCC": "Squamous Cell Carcinoma (Malignant)",
            "UNK": "Unknown / Non-lesion Artifact"
        }

    def predict_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """Executes full diagnostic pipeline on uploaded image bytes."""
        # 1. Magic Bytes Validation
        is_valid_magic, magic_err = validate_image_magic_bytes(image_bytes[:16])
        if not is_valid_magic:
            raise ValueError(f"Security validation failed: {magic_err}")
            
        # 2. Preprocess & Quality Check
        tensor, pil_img, meta = preprocess_image_bytes(image_bytes)
        
        # 3. Model Forward Pass
        mgr = ModelManager.get_instance()
        model: EfficientNetB0Classifier = mgr.model
        temp_scaler: TemperatureScaler = mgr.temp_scaler
        energy_scorer: EnergyOODScorer = mgr.energy_scorer
        device: torch.device = mgr.device

        input_batch = tensor.unsqueeze(0).to(device)
        
        with torch.inference_mode():
            logits, embedding = model(input_batch, return_embedding=True)
            
            # Calibrate Probabilities
            scaled_logits = temp_scaler(logits)
            probs = F.softmax(scaled_logits, dim=1).cpu().numpy()[0]
            
            # OOD Scoring on CPU
            energy_score = float(energy_scorer.compute_energy(logits.cpu()).item())
            is_energy_ood = bool(energy_score > energy_scorer.threshold)
            
        pred_idx = int(np.argmax(probs))
        raw_calibrated_conf = float(probs[pred_idx])
        pred_class_code = self.class_mapping[pred_idx]

        # Authentic multi-class dominance margin & Shannon entropy
        sorted_probs = np.sort(probs)[::-1]
        p_top = float(sorted_probs[0])
        p_second = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        separation_margin = float(p_top - p_second)
        normalized_entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs)))

        # 4. Generate Grad-CAM (using gradient mode)
        gradcam_engine = GradCAM(model, model.get_gradcam_target_layer())
        try:
            heatmap = gradcam_engine.generate_heatmap(input_batch, target_class=pred_idx)
            overlay_img = gradcam_engine.create_overlay(pil_img, heatmap)
        finally:
            gradcam_engine.remove_hooks()

        # 5. Persist Media Files
        file_id = str(uuid.uuid4())
        upload_filename = f"{file_id}.jpg"
        gradcam_filename = f"gc_{file_id}.jpg"

        upload_path = self.upload_dir / upload_filename
        gradcam_path = self.gradcam_dir / gradcam_filename

        pil_img.save(upload_path, format="JPEG", quality=95)
        overlay_img.save(gradcam_path, format="JPEG", quality=95)

        # 6. Build Calibrated Probabilities Map & Aggregate Risk
        prob_dict = {self.class_mapping[i]: float(probs[i]) for i in range(len(self.class_mapping))}
        
        malignant_codes = ["MEL", "BCC", "SCC", "AK"]
        benign_codes = ["NV", "BKL", "DF", "VASC"]
        malignant_risk = float(sum(prob_dict.get(c, 0.0) for c in malignant_codes))
        benign_risk = float(sum(prob_dict.get(c, 0.0) for c in benign_codes))
        
        if malignant_risk >= 0.40 or pred_class_code in ["MEL", "BCC", "SCC"]:
            triage_label = "Malignant Risk (Clinical Review Advised)"
        elif pred_class_code == "AK":
            triage_label = "Pre-cancerous Lesion"
        else:
            triage_label = "Benign Lesion Profile (Low Risk)"

        # 7. Distribution Status (Selective Prediction Gate)
        if is_energy_ood:
            status = "ood"
            display_class = "Uncertain / Outside Reliable Scope"
        else:
            status = "in_distribution"
            display_class = pred_class_code

        return {
            "status": status,
            "predicted_class": display_class,
            "full_class_name": self.class_names.get(pred_class_code, pred_class_code),
            "confidence": raw_calibrated_conf,
            "calibrated_confidence": raw_calibrated_conf,
            "diagnostic_certainty": raw_calibrated_conf,
            "separation_margin": separation_margin,
            "entropy": normalized_entropy,
            "malignant_risk": malignant_risk,
            "benign_risk": benign_risk,
            "triage_label": triage_label,
            "probabilities": prob_dict,
            "energy_score": energy_score,
            "image_url": f"/storage/uploads/{upload_filename}",
            "gradcam_url": f"/storage/gradcam/{gradcam_filename}"
        }
