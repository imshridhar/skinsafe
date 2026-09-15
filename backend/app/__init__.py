from pathlib import Path
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager

from backend.app.config import Config


def create_app(config_class=Config) -> Flask:
    """Flask Application Factory."""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize Extensions
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    JWTManager(app)
    
    # Ensure storage directories exist
    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["GRADCAM_DIR"]).mkdir(parents=True, exist_ok=True)
    
    # Register Blueprints
    from backend.app.routes.predict import predict_bp
    from backend.app.routes.auth import auth_bp
    from backend.app.routes.analytics import analytics_bp
    from backend.app.routes.history import history_bp

    app.register_blueprint(predict_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(history_bp)

    
    # Static Media Serving for Grad-CAM & Uploads
    @app.route("/storage/uploads/<path:filename>")
    def serve_upload(filename):
        return send_from_directory(app.config["UPLOAD_DIR"], filename)
        
    @app.route("/storage/gradcam/<path:filename>")
    def serve_gradcam(filename):
        return send_from_directory(app.config["GRADCAM_DIR"], filename)
        
    # Health Check
    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "online",
            "service": "isic-lesion-ai",
            "model_architecture": "efficientnet_b0",
            "version": "1.0.0"
        }), 200
        
    return app
