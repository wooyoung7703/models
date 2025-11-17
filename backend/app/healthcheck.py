import os
import sys
import asyncio

PORT = int(os.getenv("WS_PORT", "8022"))
HOST = os.getenv("WS_HOST", "127.0.0.1")

async def _run():
    import websockets
    uri = f"ws://{HOST}:{PORT}"
    try:
        # Perform a real websocket handshake; immediately close.
        async with websockets.connect(uri, ping_interval=None):
            pass
        print("OK")
        return 0
    except Exception as e:
        print(f"FAIL {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
