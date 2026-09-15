"""Backend entry point for direct execution from backend directory."""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "127.0.0.1")
    try:
        from waitress import serve
        print(f"[Waitress WSGI] Serving backend on http://{host}:{port}...")
        serve(app, host=host, port=port, threads=6)
    except ImportError:
        app.run(host=host, port=port, debug=False)
