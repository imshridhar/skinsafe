import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base application configuration."""
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-replace-in-production")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-jwt-secret-replace-in-production")
    JWT_ACCESS_TOKEN_EXPIRES = 3600 * 24  # 24 hours
    
    # MongoDB
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017/skin_lesion_ai")
    BCRYPT_ROUNDS = int(os.getenv("BCRYPT_ROUNDS", "12"))
    
    # File Storage
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "storage/uploads")
    GRADCAM_DIR = BASE_DIR / os.getenv("GRADCAM_DIR", "storage/gradcam")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
    
    # ML Model
    MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "ml/checkpoints/efficientnet_b0/best_model.pt")
    DEVICE = os.getenv("DEVICE", "cuda")
