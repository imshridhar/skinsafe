import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Multi-Class Focal Loss for handling severe class imbalance (Lin et al., ICCV 2017).

    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes exact multi-class focal loss with numerical stability.

        Args:
            logits: Predicted class logits [Batch, num_classes]
            targets: Ground-truth class indices [Batch]

        Returns:
            Computed scalar loss.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Extract true class probability & log probability
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        # Modulating factor: (1 - p_t)^gamma
        focal_weight = (1.0 - target_probs).pow(self.gamma)

        # Alpha class weighting
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(dim=0, index=targets)
            focal_weight = alpha_t * focal_weight

        # Label smoothing regularizer
        if self.label_smoothing > 0.0:
            smoothed_loss = -log_probs.mean(dim=-1)
            loss = focal_weight * (-target_log_probs) * (1.0 - self.label_smoothing) + self.label_smoothing * smoothed_loss
        else:
            loss = focal_weight * (-target_log_probs)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss



def compute_class_weights(class_counts: torch.Tensor, beta: float = 0.999) -> torch.Tensor:
    """Computes effective number of samples class weights (Class-Balanced Loss).

    Weight = (1 - beta) / (1 - beta^N_c)
    """
    effective_num = 1.0 - torch.pow(beta, class_counts.float())
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / weights.sum() * len(class_counts)
    return weights
