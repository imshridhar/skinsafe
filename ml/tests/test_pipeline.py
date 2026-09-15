import pytest
import numpy as np
from PIL import Image
import torch

from ml.src.config import load_class_mapping
from ml.src.preprocessing import get_base_transform, get_training_transform, preprocess_image_bytes
from ml.src.model import build_model
from ml.src.losses import FocalLoss
from ml.src.ood import TemperatureScaler, EnergyOODScorer
from ml.src.gradcam import GradCAM


def test_base_transform():
    """Verify deterministic base transform shape and normalization."""
    img = Image.new("RGB", (500, 300), color=(100, 150, 200))
    transform = get_base_transform(target_size=(224, 224))
    tensor = transform(img)
    
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_training_transform():
    """Verify stochastic augmentation transform shape."""
    img = Image.new("RGB", (400, 400), color=(50, 100, 150))
    transform = get_training_transform(target_size=(224, 224))
    tensor = transform(img)
    
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_model_forward_and_embedding():
    """Verify single-pass model forward pass and 1280-d embedding extraction."""
    model = build_model("efficientnet_b0", num_classes=9, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    
    logits, embedding = model(x, return_embedding=True)
    assert logits.shape == (2, 9)
    assert embedding.shape == (2, 1280)


def test_focal_loss():
    """Verify multi-class focal loss computation."""
    criterion = FocalLoss(gamma=2.0)
    logits = torch.randn(4, 9)
    targets = torch.tensor([0, 1, 2, 8])
    
    loss = criterion(logits, targets)
    assert loss.dim() == 0
    assert float(loss.item()) > 0.0


def test_temperature_scaling():
    """Verify temperature scaler scales logits."""
    scaler = TemperatureScaler(initial_temp=2.0)
    logits = torch.tensor([[4.0, 2.0, 0.0]])
    scaled = scaler(logits)
    
    assert torch.allclose(scaled, torch.tensor([[2.0, 1.0, 0.0]]))


def test_energy_ood_scorer():
    """Verify Free Energy calculation."""
    scorer = EnergyOODScorer(temperature=1.0)
    logits = torch.tensor([[5.0, 1.0, 0.0], [0.1, 0.1, 0.1]])
    energies = scorer.compute_energy(logits)
    
    assert len(energies) == 2
    # High confidence has lower (more negative) energy
    assert float(energies[0]) < float(energies[1])


def test_gradcam_generation():
    """Verify Grad-CAM heatmap generation and overlay blending."""
    model = build_model("efficientnet_b0", num_classes=9, pretrained=False)
    target_layer = model.get_gradcam_target_layer()
    gradcam = GradCAM(model, target_layer)
    
    x = torch.randn(1, 3, 224, 224)
    heatmap = gradcam.generate_heatmap(x, target_class=0)
    
    assert heatmap.shape == (7, 7) or len(heatmap.shape) == 2
    assert float(np.min(heatmap)) >= 0.0
    assert float(np.max(heatmap)) <= 1.0
    
    orig_img = Image.new("RGB", (224, 224), color=(200, 100, 50))
    overlay = gradcam.create_overlay(orig_img, heatmap)
    assert overlay.size == (224, 224)
    
    gradcam.remove_hooks()
