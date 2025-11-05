"""
Dev entrypoint to run the FastAPI server.

It ensures the repository root is on sys.path so that
``from server.main import app`` works no matter where this
script is launched from (e.g., via npm from client/).
Also loads the repo-level .env and enables reload.

Usage:
  python ../server/dev_api.py
  (or via `npm run api` in client/)
"""

from pathlib import Path
import sys
import uvicorn
from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent

# Ensure imports like `from server.main import app` resolve
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load environment variables from repo root
load_dotenv(REPO_ROOT / ".env")

# Import the ASGI app directly (after sys.path fix)
from server.main import app  # type: ignore  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=True,
        # Watch both the server directory and the repo root (for .env etc.)
        reload_dirs=[str(SERVER_DIR), str(REPO_ROOT)],
    )
