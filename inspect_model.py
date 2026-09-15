"""SkinSafe AI - Root Model Parameter Inspector Entry Point."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.src.inspect_model_params import inspect_model_parameters

if __name__ == "__main__":
    inspect_model_parameters()
