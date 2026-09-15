import pytest
from pathlib import Path
from PIL import Image
import pandas as pd

from ml.src.config import load_class_mapping, get_inv_class_mapping
from ml.src.audit_dataset import compute_sha256, inspect_image, parse_ground_truth


def test_class_mapping_completeness():
    """Verify class mapping contains all 9 canonical ISIC 2019 classes."""
    mapping = load_class_mapping("ml/configs/class_mapping.json")
    assert len(mapping) == 9
    expected_classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
    for i, cls in enumerate(expected_classes):
        assert mapping[i] == cls


def test_inv_class_mapping():
    """Verify inverse class mapping functions correctly."""
    inv_map = get_inv_class_mapping("ml/configs/class_mapping.json")
    assert inv_map["MEL"] == 0
    assert inv_map["UNK"] == 8
    assert len(inv_map) == 9


def test_inspect_image_valid(tmp_path):
    """Verify inspection on a valid synthetic RGB JPEG."""
    img_path = tmp_path / "valid.jpg"
    img = Image.new("RGB", (300, 200), color=(128, 64, 32))
    img.save(img_path, format="JPEG")
    
    result = inspect_image(img_path)
    assert result["is_valid"] is True
    assert result["validation_error"] is None
    assert result["width"] == 300
    assert result["height"] == 200
    assert result["mode"] == "RGB"
    assert len(result["sha256"]) == 64


def test_inspect_image_corrupt(tmp_path):
    """Verify inspection flags corrupted/truncated image files."""
    img_path = tmp_path / "corrupt.jpg"
    with open(img_path, "wb") as f:
        f.write(b"NOT_A_JPEG_FILE_HEADER")
        
    result = inspect_image(img_path)
    assert result["is_valid"] is False
    assert result["validation_error"] is not None


def test_parse_ground_truth(tmp_path):
    """Verify ground truth parsing handles one-hot encoded rows properly."""
    gt_csv = tmp_path / "gt.csv"
    data = {
        "image": ["ISIC_001", "ISIC_002", "ISIC_003"],
        "MEL": [1.0, 0.0, 0.0],
        "NV": [0.0, 1.0, 0.0],
        "BCC": [0.0, 0.0, 0.0],
        "AK": [0.0, 0.0, 0.0],
        "BKL": [0.0, 0.0, 0.0],
        "DF": [0.0, 0.0, 0.0],
        "VASC": [0.0, 0.0, 0.0],
        "SCC": [0.0, 0.0, 0.0],
        "UNK": [0.0, 0.0, 0.0]
    }
    pd.DataFrame(data).to_csv(gt_csv, index=False)
    
    mapping = load_class_mapping("ml/configs/class_mapping.json")
    parsed_df = parse_ground_truth(gt_csv, mapping)
    
    assert len(parsed_df) == 3
    assert parsed_df.iloc[0]["label"] == "MEL"
    assert parsed_df.iloc[1]["label"] == "NV"
    assert pd.isna(parsed_df.iloc[2]["label"])  # Sum=0 flagged as NaN
