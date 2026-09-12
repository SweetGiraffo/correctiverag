"""
CLI entry point to run the FastAPI backend server.
"""

import uvicorn
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"Starting Corrective RAG FastAPI Backend on http://{host}:{port}")
    uvicorn.run("src.api.app:app", host=host, port=port, reload=False, log_level="info")
