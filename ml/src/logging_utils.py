import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(
    name: str = "isic_platform",
    log_dir: str = "ml/reports/logs",
    run_id: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Configures structured console and file logging.

    Args:
        name: Name of the logger.
        log_dir: Path to directory where log files are stored.
        run_id: Optional unique run identifier.
        level: Logging level.

    Returns:
        Configured logging.Logger instance.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_str = f"_{run_id}" if run_id else ""
    log_filename = f"{name}_{timestamp}{run_str}.log"
    log_filepath = Path(log_dir) / log_filename
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Prevent duplicate handlers
    
    # Formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    file_handler = logging.FileHandler(str(log_filepath), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("Structured logger initialized. Log file: %s", log_filepath)
    return logger


def write_json_report(data: Dict[str, Any], filepath: str) -> None:
    """Writes a dictionary safely as formatted JSON.

    Args:
        data: Dictionary of metrics or audit records.
        filepath: Target JSON filepath.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
