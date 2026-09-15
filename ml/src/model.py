import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models

from ml.src.logging_utils import setup_logger


logger = setup_logger(name="model_factory", log_dir="ml/reports/logs")


class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 backbone with custom 9-class head and embedding extraction."""
    def __init__(
        self,
        num_classes: int = 9,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.architecture = "efficientnet_b0"
        
        # Load weights
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = models.efficientnet_b0(weights=weights)
        
        # Extract features backbone
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        # Penultimate feature dimension is 1280
        self.embedding_dim = base_model.classifier[1].in_features  # 1280
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(self.embedding_dim, num_classes)
        )
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate 1280-d embedding vector.

        Args:
            x: Input tensor [B, 3, 224, 224]

        Returns:
            Embedding tensor [B, 1280]
        """
        x = self.features(x)
        x = self.avgpool(x)
        embedding = torch.flatten(x, 1)
        return embedding

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Single-pass forward execution returning logits and optional embeddings.

        Args:
            x: Input tensor [B, 3, 224, 224]
            return_embedding: Whether to return (logits, embedding) tuple.

        Returns:
            Logits [B, num_classes] or (Logits [B, num_classes], Embedding [B, 1280])
        """
        embedding = self.extract_features(x)
        logits = self.classifier(embedding)
        
        if return_embedding:
            return logits, embedding
        return logits

    def freeze_backbone(self) -> None:
        """Freezes all backbone feature parameters for Stage A warmup training."""
        for param in self.features.parameters():
            param.requires_grad = False
        logger.info("EfficientNet-B0 backbone frozen. Only classifier head is trainable.")

    def unfreeze_top_blocks(self, n_blocks: int = 4) -> None:
        """Unfreezes the top N convolutional blocks for Stage B fine-tuning.

        Args:
            n_blocks: Number of final feature blocks to unfreeze (1 to 8).
        """
        total_blocks = len(self.features)
        start_idx = max(0, total_blocks - n_blocks)
        
        for i, block in enumerate(self.features):
            requires_grad = (i >= start_idx)
            for param in block.parameters():
                param.requires_grad = requires_grad
                
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "Unfroze top %d blocks (indices %d..%d). Trainable params: %d / %d (%.1f%%)",
            n_blocks, start_idx, total_blocks - 1, trainable, total, (trainable / total) * 100
        )

    def get_gradcam_target_layer(self) -> nn.Module:
        """Returns the final convolutional block for Grad-CAM gradient hooks."""
        return self.features[-1]


def build_model(
    architecture: str = "efficientnet_b0",
    num_classes: int = 9,
    pretrained: bool = True,
    dropout_rate: float = 0.3
) -> nn.Module:
    """Model factory supporting multiple architectures."""
    if architecture == "efficientnet_b0":
        return EfficientNetB0Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unsupported model architecture: {architecture}")
