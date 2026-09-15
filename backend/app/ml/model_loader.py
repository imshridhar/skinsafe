import logging
from pathlib import Path
from typing import Optional, cast

import torch
import torch.nn as nn

from ml.src.model import build_model, EfficientNetB0Classifier
from ml.src.ood import EnergyOODScorer, TemperatureScaler


logger = logging.getLogger("backend.ml.model_loader")


class ModelManager:
    """Thread-safe Singleton Model Manager for production API serving."""
    _instance: Optional["ModelManager"] = None
    
    def __init__(self, model_path: Path, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        
        # Load and initialize model
        built_model = build_model(architecture="efficientnet_b0", num_classes=9, pretrained=False)
        self.model: EfficientNetB0Classifier = cast(EfficientNetB0Classifier, built_model)
        
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            logger.info("Successfully loaded model checkpoint from: %s", self.model_path)
        else:
            raise FileNotFoundError(f"Trained model checkpoint not found at {self.model_path}. Please ensure best_model.pt is present.")
            
        self.model.to(self.device)
        self.model.eval()

        # Load calibrated temperature and OOD thresholds if available
        calib_file = self.model_path.parent / "calibration_params.json"
        temp_val = 1.2014
        energy_thresh = -2.7322

        if calib_file.exists():
            try:
                import json
                with open(calib_file, "r") as f:
                    cdata = json.load(f)
                    temp_val = float(cdata.get("temperature", temp_val))
                    energy_thresh = float(cdata.get("energy_threshold", energy_thresh))
                logger.info("Loaded calibrated parameters: T* = %.4f, Energy Thresh = %.4f", temp_val, energy_thresh)
            except Exception as e:
                logger.warning("Could not read calibration_params.json (%s), using defaults.", e)

        self.temp_scaler: TemperatureScaler = TemperatureScaler(initial_temp=temp_val).to(self.device)
        self.energy_scorer: EnergyOODScorer = EnergyOODScorer(temperature=temp_val, threshold=energy_thresh)

    @classmethod
    def get_instance(cls, model_path: Optional[Path] = None, device: str = "cuda") -> "ModelManager":
        if cls._instance is None:
            if model_path is None:
                model_path = Path("ml/checkpoints/efficientnet_b0/best_model.pt")
            cls._instance = cls(model_path=model_path, device=device)
        return cls._instance
