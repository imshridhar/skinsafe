from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM Visual Explanations for EfficientNet-B0 and Convolutional Backbones."""
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generates 2D normalized Grad-CAM heatmap.

        Args:
            input_tensor: Shape [1, 3, 224, 224] with requires_grad=True
            target_class: Integer class index (0..8). If None, uses argmax prediction.

        Returns:
            2D numpy array [H, W] normalized between 0.0 and 1.0.
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Ensure tensor requires gradient
        x = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(x)
        
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())
            
        score = logits[0, target_class]
        score.backward()
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")
            
        # Global average pooling on gradients: alpha_k
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])  # [C]
        
        # Weight activations by pooled gradients
        activations = self.activations[0]  # [C, H, W]
        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU
        
        # Normalize
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        else:
            heatmap = np.zeros_like(heatmap)
            
        return heatmap

    def create_overlay(
        self,
        original_img: Union[Image.Image, np.ndarray],
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET
    ) -> Image.Image:
        """Overlays heatmap on original Pillow RGB image."""
        if isinstance(original_img, Image.Image):
            orig_rgb = np.array(original_img.convert("RGB"))
        else:
            orig_rgb = original_img
            
        h, w = orig_rgb.shape[:2]
        
        # Resize heatmap to match image dimensions
        resized_heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        uint8_heatmap = np.uint8(255 * resized_heatmap)
        
        # Apply colormap
        color_heatmap = cv2.applyColorMap(uint8_heatmap, colormap)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend
        blended = (alpha * color_heatmap + (1.0 - alpha) * orig_rgb).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blended)

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
