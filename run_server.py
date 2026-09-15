"""SkinSafe AI - Production Backend Server Launcher

Runs Flask backend on http://127.0.0.1:5000 with CORS, JWT, and ML Inference Engine.
Supports Waitress multi-threaded WSGI serving or standard Flask development server.
"""

import os
import sys
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "127.0.0.1")
    
    print("\n" + "=" * 65)
    print("  SkinSafe AI Research Platform - Backend Inference Server")
    print(f"  Listening on: http://{host}:{port}")
    print("  Health Check: http://127.0.0.1:5000/api/health")
    print("  API Docs:     ISIC 2019 EfficientNet-B0 + OOD + Grad-CAM")
    print("=" * 65 + "\n")
    
    try:
        from waitress import serve
        print(f"[Waitress WSGI] Serving multi-threaded production server on port {port}...")
        serve(app, host=host, port=port, threads=6)
    except ImportError:
        print(f"[Flask Server] Serving development server on port {port}...")
        app.run(host=host, port=port, debug=False)
