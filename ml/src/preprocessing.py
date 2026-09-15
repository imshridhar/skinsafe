import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from ml.src.config import load_yaml_config, load_class_mapping


class ImageQualityValidator:
    """Validates raw image quality to detect blank, underexposed, or corrupt images."""
    def __init__(
        self,
        min_width: int = 100,
        min_height: int = 100,
        min_brightness: float = 10.0,
        max_brightness: float = 245.0,
        min_contrast: float = 8.0
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast

    def validate(self, img: Image.Image) -> Tuple[bool, Optional[str]]:
        w, h = img.size
        if w < self.min_width or h < self.min_height:
            return False, f"Image dimensions too small: ({w}x{h})"
            
        # Convert to grayscale numpy for luminance statistics
        gray = np.array(img.convert("L"), dtype=np.float32)
        mean_lum = float(np.mean(gray))
        std_lum = float(np.std(gray))
        
        if mean_lum < self.min_brightness:
            return False, f"Severely underexposed / dark image (mean brightness: {mean_lum:.1f})"
        if mean_lum > self.max_brightness:
            return False, f"Severely overexposed / washed out image (mean brightness: {mean_lum:.1f})"
        if std_lum < self.min_contrast:
            return False, f"Extremely low contrast / blank image (contrast std: {std_lum:.1f})"
            
        return True, None


def get_base_transform(
    target_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> T.Compose:
    """Returns deterministic transform pipeline for validation, test, and production API."""
    return T.Compose([
        T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def get_training_transform(
    target_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> T.Compose:
    """Returns training-only stochastic augmentation transform pipeline with Cutout/RandomErasing."""
    return T.Compose([
        T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=20, interpolation=T.InterpolationMode.BICUBIC),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
        T.RandomErasing(p=0.20, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random")
    ])


def predict_with_tta(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Executes Test-Time Augmentation (TTA) averaging original, horizontal, and vertical flip views.

    Args:
        model: PyTorch classification model in eval mode.
        images: Input batch tensor [B, 3, 224, 224]

    Returns:
        Averaged ensemble softmax probability tensor [B, num_classes].
    """
    with torch.inference_mode():
        # 1. Original
        p1 = torch.softmax(model(images), dim=1)
        # 2. Horizontal Flip
        p2 = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
        # 3. Vertical Flip
        p3 = torch.softmax(model(torch.flip(images, dims=[2])), dim=1)
        # 4. Diagonal Flip (Both H & V)
        p4 = torch.softmax(model(torch.flip(images, dims=[2, 3])), dim=1)

    return (p1 + p2 + p3 + p4) / 4.0



def preprocess_image_bytes(
    image_bytes: bytes,
    transform: Optional[Callable] = None,
    validator: Optional[ImageQualityValidator] = None
) -> Tuple[torch.Tensor, Image.Image, Dict[str, Any]]:
    """Preprocesses raw image bytes for API inference or offline evaluation.

    Args:
        image_bytes: Raw bytes from upload or file read.
        transform: Optional transform. Defaults to base deterministic transform.
        validator: Optional quality validator.

    Returns:
        Tuple of (Tensor [3, 224, 224], Pillow RGB Image, Metadata Dict).
    """
    if transform is None:
        transform = get_base_transform()
    if validator is None:
        validator = ImageQualityValidator()
        
    img = Image.open(io.BytesIO(image_bytes))
    img.load()  # Force load bytes
    
    # Standardize orientation if EXIF present
    img = ImageOps.exif_transpose(img)
    
    # Convert to standard RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    is_valid, error = validator.validate(img)
    if not is_valid:
        raise ValueError(f"Image quality validation failed: {error}")
        
    orig_size = img.size
    tensor = transform(img)
    
    metadata = {
        "original_width": orig_size[0],
        "original_height": orig_size[1],
        "tensor_shape": list(tensor.shape),
        "mode": img.mode
    }
    
    return tensor, img, metadata


class ISICDataset(Dataset):
    """PyTorch Dataset for ISIC 2019 split CSVs."""
    def __init__(
        self,
        csv_path: Union[str, Path],
        root_dir: Union[str, Path] = ".",
        transform: Optional[Callable] = None,
        is_training: bool = False
    ):
        self.df = pd.read_csv(csv_path)
        self.root_dir = Path(root_dir)
        self.is_training = is_training
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_training_transform() if is_training else get_base_transform()
            
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = self.root_dir / str(row["image_path"])
        
        with Image.open(img_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensor = self.transform(img)
            
        label_idx = int(row["class_index"]) if pd.notna(row.get("class_index")) else -1
        label_code = str(row["label"]) if pd.notna(row.get("label")) else "UNK"
        
        return {
            "image": tensor,
            "label_idx": torch.tensor(label_idx, dtype=torch.long),
            "label_code": label_code,
            "image_id": str(row["image_id"])
        }
