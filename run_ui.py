"""
CLI entry point to run the Streamlit frontend.
Automatically finds an available port if 8501 is already in use.
"""

import subprocess
import sys
import os
import socket
import argparse


def find_free_port(start_port: int = 8501, max_tries: int = 50) -> int:
    """Finds an open port starting from start_port."""
    for p in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start_port


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Streamlit UI")
    parser.add_argument("--port", type=int, default=None, help="Explicit port to run Streamlit on")
    args, unknown = parser.parse_known_args()

    app_path = os.path.join(os.path.dirname(__file__), "frontend", "app.py")

    # Determine port
    if args.port:
        port = args.port
    elif os.environ.get("STREAMLIT_PORT"):
        port = int(os.environ["STREAMLIT_PORT"])
    else:
        port = find_free_port(8501)

    if port != 8501 and not args.port and not os.environ.get("STREAMLIT_PORT"):
        print(f"[Notice] Port 8501 is occupied. Auto-switching to available port: {port}")

    print(f"Starting Streamlit UI on http://localhost:{port}...")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(port),
        "--server.headless",
        "true"
    ]
    subprocess.run(cmd)
