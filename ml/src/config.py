import json
from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads and validates a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Dictionary containing the parsed configuration.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_class_mapping(mapping_path: str = "ml/configs/class_mapping.json") -> Dict[int, str]:
    """Loads the canonical 9-class mapping.

    Args:
        mapping_path: Path to the JSON class mapping file.

    Returns:
        Dictionary mapping integer class indices (0..8) to string class codes.
    """
    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"Class mapping file not found: {mapping_path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def get_inv_class_mapping(mapping_path: str = "ml/configs/class_mapping.json") -> Dict[str, int]:
    """Returns inverted mapping from class code string to integer index."""
    direct = load_class_mapping(mapping_path)
    return {v: k for k, v in direct.items()}
