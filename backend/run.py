import os
import socket
import uvicorn

# Import FastAPI app with flexible path (supports running from repo root or backend folder)
try:
    from app.main import app  # when CWD is backend/
except ImportError:  # running as module: python -m backend.run from repo root
    from backend.app.main import app


def is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def find_free_port(host: str = "127.0.0.1", start: int = 8022, end: int = 8040) -> int | None:
    for p in range(start, end + 1):
        if is_port_free(host, p):
            return p
    return None


def main():
    host = os.getenv("HOST", "127.0.0.1")
    port_env = os.getenv("PORT") or os.getenv("APP_PORT")
    if port_env:
        try:
            port = int(port_env)
        except ValueError:
            port = None
    else:
        port = None

    if port is None:
        port = find_free_port(host, 8022, 8040) or 8022

    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
