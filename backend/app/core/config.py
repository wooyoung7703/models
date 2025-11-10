import os
from pathlib import Path
from dotenv import load_dotenv

# Load backend/.env by default. Use override=True to reflect changes during dev.
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / '.env', override=True)


class Settings:
    # Database
    DB_URL: str = os.getenv(
        "DB_URL",
        # default sqlite file under backend/data
        f"sqlite:///{Path(__file__).resolve().parent.parent.parent / 'data' / 'data.db'}",
    )

    # Binance
    EXCHANGE_TYPE: str = os.getenv("EXCHANGE_TYPE", "spot")  # 'spot' or 'futures'
    SYMBOL: str = os.getenv("SYMBOL", "xrpusdt").lower()
    # Optional: comma-separated list for multi-symbol mode. If not provided, defaults to [SYMBOL]
    SYMBOLS_RAW: str = os.getenv("SYMBOLS", "")
    @property
    def SYMBOLS(self):
        if getattr(self, "_SYMBOLS_CACHE", None) is not None:
            return self._SYMBOLS_CACHE
        if self.SYMBOLS_RAW.strip():
            syms = [s.strip().lower() for s in self.SYMBOLS_RAW.split(",") if s.strip()]
        else:
            syms = [self.SYMBOL]
        self._SYMBOLS_CACHE = syms
        return syms
    INTERVAL: str = os.getenv("INTERVAL", "1m")
    # Spot and Futures base URLs
    BINANCE_WS_BASE_SPOT: str = os.getenv(
        "BINANCE_WS_BASE_SPOT", "wss://stream.binance.com:9443/ws"
    )
    BINANCE_WS_BASE_FUTURES: str = os.getenv(
        "BINANCE_WS_BASE_FUTURES", "wss://fstream.binance.com/ws"
    )
    # Combined stream endpoints for multi-symbol
    BINANCE_WS_COMBINED_SPOT: str = os.getenv(
        "BINANCE_WS_COMBINED_SPOT", "wss://stream.binance.com:9443/stream"
    )
    BINANCE_WS_COMBINED_FUTURES: str = os.getenv(
        "BINANCE_WS_COMBINED_FUTURES", "wss://fstream.binance.com/stream"
    )

    RECONNECT_MIN_SEC: float = float(os.getenv("RECONNECT_MIN_SEC", "1"))
    RECONNECT_MAX_SEC: float = float(os.getenv("RECONNECT_MAX_SEC", "30"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json").lower()  # 'json' or 'plain'
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "models-backend")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "dev")

    # Gap fill on startup
    GAP_FILL_ENABLED: bool = os.getenv("GAP_FILL_ENABLED", "1") in {"1", "true", "True"}
    # How far back (in minutes) to scan and fill forward from if DB empty or gaps exist
    GAP_FILL_LOOKBACK_MINUTES: int = int(os.getenv("GAP_FILL_LOOKBACK_MINUTES", "1440"))  # default 1 day

    # Background tasks controls
    # Disable long-running background loops (collector/resampler/predictor) for tests or special runs
    DISABLE_BACKGROUND_LOOPS: bool = os.getenv("DISABLE_BACKGROUND_LOOPS", "0") in {"1", "true", "True"}
    # Enable/disable periodic prediction task
    PREDICT_ENABLED: bool = os.getenv("PREDICT_ENABLED", "1") in {"1", "true", "True"}
    # Prediction tick interval seconds
    PREDICT_INTERVAL_SECONDS: int = int(os.getenv("PREDICT_INTERVAL_SECONDS", "30"))


settings = Settings()
