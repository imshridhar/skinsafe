import logging
from flask import Blueprint, current_app, jsonify, request
from backend.app.ml.inference_service import InferenceService

predict_bp = Blueprint("predict", __name__, url_prefix="/api/v1")
logger = logging.getLogger("backend.routes.predict")


@predict_bp.route("/predict", methods=["POST"])
def predict():
    """Diagnostic Image Inference Endpoint."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided in request. Key must be 'image'"}), 400
        
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename provided"}), 400
        
    try:
        image_bytes = file.read()
        service = InferenceService(
            upload_dir=current_app.config["UPLOAD_DIR"],
            gradcam_dir=current_app.config["GRADCAM_DIR"]
        )
        result = service.predict_image_bytes(image_bytes)
        return jsonify(result), 200
        
    except ValueError as e:
        logger.warning("Validation error on image upload: %s", str(e))
        return jsonify({"error": str(e), "category": "validation_error"}), 422
    except Exception as e:
        logger.exception("Unexpected error during image inference")
        return jsonify({"error": "Internal server processing error", "details": str(e)}), 500
