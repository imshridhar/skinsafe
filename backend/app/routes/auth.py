import logging
import bcrypt
from flask import Blueprint, current_app, jsonify, request
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity

from backend.app.db import get_database
from backend.app.utils.security import validate_password_strength

auth_bp = Blueprint("auth", __name__, url_prefix="/api/v1/auth")
logger = logging.getLogger("backend.routes.auth")

# Resilient in-memory fallback user registry if MongoDB is in offline mode
_LOCAL_USERS = {
    "doctor@skinsafe.ai": {
        "email": "doctor@skinsafe.ai",
        "password_hash": bcrypt.hashpw("SkinSafe#2026".encode("utf-8"), bcrypt.gensalt(10)).decode("utf-8"),
        "role": "Chief Dermatologist",
        "name": "Dr. Sarah Jenkins, MD"
    }
}


@auth_bp.route("/signup", methods=["POST"])
def signup():
    """User Registration Endpoint with strict password entropy validation."""
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    confirm_password = data.get("confirm_password", "")
    name = data.get("name", "Clinician Researcher")
    
    if not email or "@" not in email:
        return jsonify({"error": "Valid email address is required"}), 400
        
    if password != confirm_password:
        return jsonify({"error": "Password and confirmation password do not match"}), 400
        
    is_valid_pw, pw_err = validate_password_strength(password)
    if not is_valid_pw:
        return jsonify({"error": pw_err}), 422
        
    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=current_app.config.get("BCRYPT_ROUNDS", 10))).decode("utf-8")
    
    db = get_database()
    if db is not None:
        try:
            users_col = db["users"]
            if users_col.find_one({"email": email}):
                return jsonify({"error": "Email is already registered"}), 409
                
            users_col.insert_one({
                "email": email,
                "password_hash": hashed_pw,
                "role": "researcher",
                "name": name
            })
        except Exception as e:
            logger.warning("MongoDB write failed, falling back to memory store: %s", e)
            _LOCAL_USERS[email] = {"email": email, "password_hash": hashed_pw, "role": "researcher", "name": name}
    else:
        if email in _LOCAL_USERS:
            return jsonify({"error": "Email is already registered"}), 409
        _LOCAL_USERS[email] = {"email": email, "password_hash": hashed_pw, "role": "researcher", "name": name}
        
    token = create_access_token(identity=email)
    return jsonify({
        "message": "User registered successfully",
        "access_token": token,
        "email": email,
        "name": name,
        "role": "Dermatology Researcher"
    }), 201


@auth_bp.route("/signin", methods=["POST"])
def signin():
    """User Authentication Endpoint."""
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
        
    user = None
    db = get_database()
    if db is not None:
        try:
            user = db["users"].find_one({"email": email})
        except Exception as e:
            logger.warning("MongoDB query failed, falling back to memory store: %s", e)
            user = _LOCAL_USERS.get(email)
    else:
        user = _LOCAL_USERS.get(email)
        
    if not user or not bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        return jsonify({"error": "Invalid email or password"}), 401
        
    token = create_access_token(identity=email)
    return jsonify({
        "message": "Authentication successful",
        "access_token": token,
        "email": email,
        "name": user.get("name", "Dr. Clinician"),
        "role": user.get("role", "Dermatology Specialist")
    }), 200


@auth_bp.route("/me", methods=["GET"])
@jwt_required(optional=True)
def get_current_user():
    """Returns currently authenticated user profile."""
    identity = get_jwt_identity()
    if not identity:
        return jsonify({"authenticated": False, "user": None}), 200
        
    db = get_database()
    user = None
    if db is not None:
        try:
            user = db["users"].find_one({"email": identity})
        except Exception:
            pass
            
    if not user:
        user = _LOCAL_USERS.get(identity, {"email": identity, "role": "Dermatology Specialist", "name": "Dr. Clinician"})

    return jsonify({
        "authenticated": True,
        "user": {
            "email": identity,
            "role": user.get("role", "Dermatology Specialist"),
            "name": user.get("name", "Dr. Clinician")
        }
    }), 200
