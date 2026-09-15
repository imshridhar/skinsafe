import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.covariance import LedoitWolf

from ml.src.config import load_yaml_config
from ml.src.logging_utils import setup_logger, write_json_report


logger = setup_logger(name="ood_pipeline", log_dir="ml/reports/logs")


class TemperatureScaler(nn.Module):
    """Calibrates model logits using Temperature Scaling (Guo et al., 2017).

    Finds optimal scalar T* by minimizing Negative Log-Likelihood (NLL) on held-out calibration set.
    """
    def __init__(self, initial_temp: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by temperature: logits / T."""
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> float:
        """Optimizes temperature parameter T on calibration data."""
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        optimal_t = float(self.temperature.item())
        logger.info("Temperature scaling optimization converged. Optimal T* = %.4f", optimal_t)
        return optimal_t


class EnergyOODScorer:
    """Free Energy-based Out-Of-Distribution detector (Liu et al., NeurIPS 2020).

    E(x; T) = -T * log(sum(exp(f_i(x) / T)))
    """
    def __init__(self, temperature: float = 1.0, threshold: float = -10.0):
        self.temperature = temperature
        self.threshold = threshold

    def compute_energy(self, logits: torch.Tensor) -> torch.Tensor:
        """Computes free energy scores for a batch of logits."""
        # Energy = -T * LogSumExp(logits / T)
        scaled_logits = logits / self.temperature
        energy = -self.temperature * torch.logsumexp(scaled_logits, dim=1)
        return energy

    def fit_threshold(self, calib_logits: torch.Tensor, target_tpr: float = 0.95) -> float:
        """Sets OOD threshold to achieve target TPR (e.g. 95%) on calibration data."""
        energies = self.compute_energy(calib_logits).cpu().numpy()
        # In-distribution has low energy (more negative), so threshold is 95th percentile
        self.threshold = float(np.percentile(energies, target_tpr * 100))
        logger.info("Energy OOD threshold set to: %.4f (at %.1f%% TPR)", self.threshold, target_tpr * 100)
        return self.threshold

    def is_ood(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns boolean tensor where True = Out-Of-Distribution."""
        energies = self.compute_energy(logits)
        return energies > self.threshold


class MahalanobisOODScorer:
    """Mahalanobis Distance OOD scorer on penultimate 1280-d feature embeddings."""
    def __init__(self, embedding_dim: int = 1280, num_classes: int = 9):
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.class_centroids: Optional[np.ndarray] = None
        self.precision_matrix: Optional[np.ndarray] = None
        self.threshold: float = 100.0

    def fit(self, embeddings: np.ndarray, labels: np.ndarray, target_tpr: float = 0.95) -> None:
        """Computes class centroids and tied empirical/shrinkage covariance matrix."""
        logger.info("Fitting Mahalanobis OOD scorer on %d embeddings...", len(embeddings))
        self.class_centroids = np.zeros((self.num_classes, self.embedding_dim))
        
        # Calculate centroids
        for c in range(self.num_classes):
            mask = (labels == c)
            if np.sum(mask) > 0:
                self.class_centroids[c] = np.mean(embeddings[mask], axis=0)
            else:
                self.class_centroids[c] = np.zeros(self.embedding_dim)
                
        # Centered residuals for tied covariance
        residuals = []
        for i in range(len(embeddings)):
            c = labels[i]
            residuals.append(embeddings[i] - self.class_centroids[c])
        residuals = np.array(residuals)
        
        # Ledoit-Wolf shrinkage for numerical stability
        lw = LedoitWolf()
        lw.fit(residuals)
        self.precision_matrix = lw.precision_
        
        # Compute distances on calibration data to find threshold
        distances = [self.compute_distance(emb) for emb in embeddings]
        self.threshold = float(np.percentile(distances, target_tpr * 100))
        logger.info("Mahalanobis OOD threshold set to: %.4f (at %.1f%% TPR)", self.threshold, target_tpr * 100)

    def compute_distance(self, embedding: np.ndarray) -> float:
        """Calculates minimum Mahalanobis distance to any class centroid."""
        if self.class_centroids is None or self.precision_matrix is None:
            raise RuntimeError("Mahalanobis scorer not fitted.")
            
        min_dist = float("inf")
        for c in range(self.num_classes):
            diff = embedding - self.class_centroids[c]
            dist = float(np.sqrt(np.dot(np.dot(diff, self.precision_matrix), diff.T)))
            if dist < min_dist:
                min_dist = dist
        return min_dist


class SelectivePredictionEngine:
    """Distribution & Reliability Engine evaluating in-distribution vs OOD status."""
    def __init__(
        self,
        temperature_scaler: TemperatureScaler,
        energy_scorer: EnergyOODScorer,
        mahalanobis_scorer: Optional[MahalanobisOODScorer] = None,
    ):
        self.temp_scaler = temperature_scaler
        self.energy_scorer = energy_scorer
        self.mahala_scorer = mahalanobis_scorer

    def evaluate_sample(
        self,
        raw_logits: torch.Tensor,
        embedding: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Evaluates single sample for calibrated prediction, OOD, and uncertainty."""
        scaled_logits = self.temp_scaler(raw_logits)
        probs = F.softmax(scaled_logits, dim=1).cpu().numpy()[0]
        
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        
        # Normalized entropy
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs)))
        
        # OOD checks
        energy_val = float(self.energy_scorer.compute_energy(raw_logits).cpu().item())
        is_energy_ood = bool(energy_val > self.energy_scorer.threshold)
        
        is_mahala_ood = False
        mahala_dist = 0.0
        if self.mahala_scorer is not None and embedding is not None:
            mahala_dist = self.mahala_scorer.compute_distance(embedding)
            is_mahala_ood = bool(mahala_dist > self.mahala_scorer.threshold)
            
        # Distribution Status
        if is_energy_ood or is_mahala_ood:
            status = "ood"
            predicted_class = None
        else:
            status = "in_distribution"
            predicted_class = pred_idx
            
        return {
            "status": status,
            "predicted_class_index": predicted_class,
            "confidence": confidence,
            "entropy": entropy,
            "probabilities": probs.tolist(),
            "energy_score": energy_val,
            "mahalanobis_distance": mahala_dist
        }
