from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

try:
    from pydantic import Field, computed_field, model_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError as exc:  # pragma: no cover - helps users install dependency
    raise ImportError(
        "pydantic-settings is required for backend.app.core.config; install it via pip."
    ) from exc


# Load both project root .env (if present) then backend/.env so backend can override.
_CONFIG_FILE = Path(__file__).resolve()
_BACKEND_DIR = _CONFIG_FILE.parents[2]  # .../backend
_ROOT_DIR = _CONFIG_FILE.parents[3]     # repo root
_DATA_DIR = _BACKEND_DIR / "data"
_MODELS_DIR = _BACKEND_DIR / "app" / "training" / "models"
_ROOT_DATA_DIR = _ROOT_DIR / "data"
root_env = _ROOT_DIR / '.env'
backend_env = _BACKEND_DIR / '.env'
if root_env.exists():
    load_dotenv(dotenv_path=root_env, override=False)
if backend_env.exists():
    # Do NOT override real environment variables with backend/.env.
    # This ensures values provided by the running process (e.g., CI, shell) take precedence.
    load_dotenv(dotenv_path=backend_env, override=False)


class Settings(BaseSettings):
    """Central configuration parsed from environment variables."""

    model_config = SettingsConfigDict(extra="ignore")

    # Database
    DB_URL: str = Field(
        default=f"sqlite:///{(_BACKEND_DIR / 'data' / 'data.db').resolve()}",
    )

    # Binance
    EXCHANGE_TYPE: str = Field(default="spot")
    SYMBOL: str = Field(default="xrpusdt")
    SYMBOLS_RAW: str = Field(default="", validation_alias="SYMBOLS")
    INTERVAL: str = Field(default="1m")
    BINANCE_WS_BASE_SPOT: str = Field(
        default="wss://stream.binance.com:9443/ws",
    )
    BINANCE_WS_BASE_FUTURES: str = Field(
        default="wss://fstream.binance.com/ws",
    )
    BINANCE_WS_COMBINED_SPOT: str = Field(
        default="wss://stream.binance.com:9443/stream",
    )
    BINANCE_WS_COMBINED_FUTURES: str = Field(
        default="wss://fstream.binance.com/stream",
    )
    RECONNECT_MIN_SEC: float = Field(default=1.0)
    RECONNECT_MAX_SEC: float = Field(default=30.0)
    WS_OPEN_TIMEOUT_SECONDS: int = Field(default=15)

    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="json")
    SERVICE_NAME: str = Field(default="models-backend")
    ENVIRONMENT: str = Field(default="dev")

    # Gap fill on startup
    GAP_FILL_ENABLED: bool = Field(default=True)
    GAP_FILL_LOOKBACK_MINUTES: int = Field(
        default=1440,
    )

    # Background tasks controls
    DISABLE_BACKGROUND_LOOPS: bool = Field(default=False)
    PREDICT_ENABLED: bool = Field(default=True)
    PREDICT_INTERVAL_SECONDS: int = Field(default=10)
    PREDICT_QUEUE_MAXSIZE: int = Field(default=8)
    PREDICT_QUEUE_GET_TIMEOUT: float = Field(default=0.5)
    PREDICT_QUEUE_IDLE_SLEEP_SECONDS: float = Field(
        default=0.05,
    )
    RESAMPLER_INTERVAL_SECONDS: int = Field(default=55)
    FEATURE_SNAPSHOT_EVERY: int = Field(default=50)
    FAST_FEATURES_ENABLED: bool = Field(default=True)

    # Shared training window for bottom-detection models (days of history)
    BOTTOM_TRAIN_DAYS: int = Field(default=14)

    # Alerts & error throttling
    ALERT_WEBHOOK_URL: Optional[str] = Field(default=None)
    ALERT_COOLDOWN_SECONDS: int = Field(default=300)
    ERROR_LOG_MIN_INTERVAL_SECONDS: int = Field(
        default=60,
    )

    # --- Model integration settings ---
    MODEL_TYPE: str = Field(default="heuristic")
    ENABLE_BASE_MODELS: bool = Field(default=True)
    ENABLE_STACKING: bool = Field(default=True)
    MODEL_XGB_PATH: str = Field(
        default=str(_MODELS_DIR / "xgb_bottom_real_1m_30d.pkl"),
    )
    MODEL_XGB_FEATURES_PATH: str = Field(
        default=str(_MODELS_DIR / "xgb_features.json"),
    )
    MODEL_LSTM_PATH: str = Field(
        default=str(_MODELS_DIR / "lstm_model.pt"),
    )
    MODEL_TRANSFORMER_PATH: str = Field(
        default=str(_MODELS_DIR / "transformer_model.pt"),
    )
    STACKING_META_PATH: str = Field(
        default=str(_MODELS_DIR / "stacking_meta.json"),
    )
    SEQ_BUFFER_STATE_PATH: str = Field(
        default=str(_DATA_DIR / "seq_buffer_state.json"),
    )
    SEQ_BUFFER_SNAPSHOT_COMPRESS: bool = Field(
        default=True,
    )
    SEQ_BUFFER_SNAPSHOT_ARCHIVE_DIR: str = Field(
        default=str(_DATA_DIR / "seq_buffer_archives"),
    )
    SEQ_BUFFER_SNAPSHOT_ARCHIVE_KEEP: int = Field(
        default=48,
    )
    STACKING_THRESHOLD: float = Field(default=0.78)
    STACKING_OVERRIDE_METHOD: str = Field(default="")
    STACKING_INCLUDE_ALL_READY: bool = Field(
        default=True,
    )
    SEQ_LEN: int = Field(default=30)
    SEQ_MIN_READY: int = Field(default=10)
    ADD_COOLDOWN_SECONDS: int = Field(default=600)
    TAKE_PROFIT_PCT: float = Field(default=0.008)
    STOP_LOSS_PCT_RAW: Optional[str] = Field(
        default=None,
        validation_alias="STOP_LOSS_PCT",
    )
    TP_MODE: str = Field(default="fixed")
    TP_TRIGGER: float = Field(default=0.005)
    TP_STEP: float = Field(default=0.0005)
    TP_GIVEBACK: float = Field(default=0.001)

    # Fees
    FEE_ENTRY_PCT: float = Field(default=0.0004)
    FEE_EXIT_PCT: float = Field(default=0.0004)

    # Backtest/Exit policy toggles
    DISABLE_STOP_LOSS: bool = Field(default=True)
    TP_DECISION_ON_NET: bool = Field(default=True)

    # Calibration & smoothing
    ENABLE_CALIBRATION: bool = Field(default=True)
    CALIBRATION_METHOD: str = Field(default="platt")
    ENABLE_PROB_SMOOTHING: bool = Field(default=True)
    PROB_EMA_ALPHA: float = Field(default=0.3)
    ENABLE_ADAPTIVE_THRESHOLD: bool = Field(
        default=True,
    )
    ADAPTIVE_THRESHOLD_QUANTILE: float = Field(
        default=0.98,
    )
    ADAPTIVE_MIN_HISTORY: int = Field(default=120)
    ADAPTIVE_HISTORY_MAX: int = Field(default=2000)
    ADAPTIVE_THRESHOLD_CLAMP_LOW: float = Field(
        default=0.50,
    )
    ADAPTIVE_THRESHOLD_CLAMP_HIGH: float = Field(
        default=0.995,
    )

    # Label schema versioning
    LABEL_SCHEMA_VERSION: str = Field(default="bottom_v1")

    # Adaptive volatility labeling
    ENABLE_VOL_LABELING: bool = Field(default=False)
    VOL_LABEL_ATR_FEATURE: str = Field(default="atr_14")
    VOL_LABEL_BASE_ATR_PCT: float = Field(default=0.01)
    VOL_LABEL_MIN_SCALE: float = Field(default=0.5)
    VOL_LABEL_MAX_SCALE: float = Field(default=2.0)

    # AutoEncoder feature augmentation
    AE_AUGMENT: bool = Field(default=False)
    AE_MODEL_PATH: str = Field(
        default=str(_MODELS_DIR / "ae_timestep.pt"),
    )

    # Probability drift monitoring
    PROB_DRIFT_ENABLED: bool = Field(default=False)
    PROB_DRIFT_RECENT_SIZE: int = Field(default=500)
    PROB_DRIFT_MIN_BASELINE: int = Field(default=50)
    PROB_DRIFT_KS_P_THRESHOLD: float = Field(
        default=0.01,
    )
    PROB_DRIFT_WASSERSTEIN_THRESHOLD: float = Field(
        default=0.15,
    )
    PROB_DRIFT_MIN_INTERVAL_HOURS: float = Field(
        default=12,
    )
    PROB_DRIFT_CONSECUTIVE_REQUIRED: int = Field(
        default=2,
    )

    # Event-driven retraining
    RETRAIN_ON_TRADE_CLOSE: bool = Field(default=False)
    RETRAIN_MIN_INTERVAL_HOURS: float = Field(
        default=12,
    )
    STACKING_META_DAYS: int = Field(default=45)
    RETRAIN_DAYS: int = Field(default=14)
    RETRAIN_STACKING_DAYS_RAW: Optional[str] = Field(
        default=None,
        validation_alias="RETRAIN_STACKING_DAYS",
    )

    # Model file watcher
    MODEL_WATCH_ENABLED: bool = Field(default=True)
    MODEL_WATCH_INTERVAL_SECONDS: int = Field(
        default=60,
    )

    # Bottom vs Forecast meta
    BOTTOM_VS_FORECAST_META_PATH: str = Field(
        default=str(_ROOT_DATA_DIR / "bottom_vs_forecast_meta.json"),
    )
    FORECAST_HORIZON: int = Field(default=12)
    FORECAST_ATR_ALPHA: float = Field(default=0.2)
    ADJUSTED_DELTA_CLAMP: float = Field(default=0.25)
    ADJUSTED_DELTA_FALLBACK_MODE: str = Field(
        default="limit",
    )
    ADJUSTED_DIVERGENCE_MEAN_THRESHOLD: float = Field(
        default=0.15,
    )
    ADJUSTED_DIVERGENCE_ABS_MEAN_THRESHOLD: float = Field(
        default=0.18,
    )
    ADJUSTED_DIVERGENCE_MIN_SAMPLES: int = Field(
        default=60,
    )
    ADJUSTED_DIVERGENCE_ACTION: str = Field(
        default="revert",
    )

    # Automated meta retraining
    META_RETRAIN_ENABLED: bool = Field(default=False)
    META_RETRAIN_EVERY_MINUTES: int = Field(
        default=0,
    )
    META_RETRAIN_DAILY_AT: str = Field(default="03:15")
    META_RETRAIN_MIN_INTERVAL_MINUTES: int = Field(
        default=60,
    )
    META_RETRAIN_MIN_REL_BRIER_IMPROVE: float = Field(
        default=0.005,
    )
    BOTTOM_VS_FORECAST_EVAL_CSV_PATH: str = Field(
        default=str(_ROOT_DATA_DIR / "bottom_eval_sample.csv"),
    )
    BOTTOM_VS_FORECAST_META_OUT_PATH: str = Field(
        default=str(_ROOT_DATA_DIR / "bottom_vs_forecast_meta.json"),
    )

    # Entry Meta
    ENABLE_ENTRY_META: bool = Field(default=False)
    ENTRY_META_PATH: str = Field(
        default=str(_ROOT_DATA_DIR / "entry_meta.json"),
    )
    ENTRY_META_THRESHOLD: float = Field(default=0.90)
    ENTRY_META_USE_ADJUSTED_PROB: bool = Field(
        default=True,
    )
    ENTRY_META_MIN_INTERVAL_SECONDS: int = Field(
        default=0,
    )
    ENTRY_META_GATE_ENABLED: bool = Field(
        default=True,
    )
    ENTRY_WINRATE_WINDOW: int = Field(default=30)
    ENTRY_WINRATE_MIN_SAMPLES: int = Field(default=10)
    ENTRY_WINRATE_TARGET: float = Field(default=0.75)
    ENTRY_SESSION_SPLIT_ENABLED: bool = Field(
        default=False,
    )
    ENTRY_SESSION_DAY_START_HOUR: int = Field(
        default=9,
    )
    ENTRY_SESSION_DAY_END_HOUR: int = Field(
        default=21,
    )
    ENTRY_WINRATE_TARGET_DAY: Optional[float] = Field(
        default=None,
    )
    ENTRY_WINRATE_TARGET_NIGHT: Optional[float] = Field(
        default=None,
    )
    ENTRY_META_DYNAMIC_STEP: float = Field(default=0.01)
    ENTRY_META_DYNAMIC_MIN: float = Field(default=0.85)
    ENTRY_META_DYNAMIC_MAX: float = Field(default=0.98)
    ENTRY_META_ONBOARD_SAMPLES: Optional[int] = Field(
        default=None,
    )
    ENTRY_META_ONBOARD_STEP_SCALE: float = Field(
        default=0.5,
    )
    ENTRY_META_STATE_PATH: str = Field(
        default=str(_DATA_DIR / "entry_meta_state.json"),
    )
    TRADE_STRATEGY: str = Field(default="stacking")
    HEURISTIC_ENTRY_THRESHOLD: float = Field(default=0.65)
    HEURISTIC_ADD_THRESHOLD: float = Field(default=0.6)

    @computed_field(return_type=List[str])
    @property
    def SYMBOLS(self) -> List[str]:
        if self.SYMBOLS_RAW and self.SYMBOLS_RAW.strip():
            return [s.strip().lower() for s in self.SYMBOLS_RAW.split(",") if s.strip()]
        return [self.SYMBOL]

    @computed_field(return_type=Optional[float])
    @property
    def STOP_LOSS_PCT(self) -> Optional[float]:
        if self.STOP_LOSS_PCT_RAW is None:
            return None
        try:
            return float(self.STOP_LOSS_PCT_RAW)
        except (TypeError, ValueError):
            return None

    @computed_field(return_type=int)
    @property
    def RETRAIN_STACKING_DAYS(self) -> int:
        if self.RETRAIN_STACKING_DAYS_RAW and self.RETRAIN_STACKING_DAYS_RAW.strip():
            try:
                return int(self.RETRAIN_STACKING_DAYS_RAW)
            except (TypeError, ValueError):
                pass
        return self.STACKING_META_DAYS

    @model_validator(mode="after")
    def _normalize_fields(self) -> "Settings":
        self.SYMBOL = self.SYMBOL.lower()
        self.LOG_FORMAT = self.LOG_FORMAT.lower()
        self.LOG_LEVEL = self.LOG_LEVEL.upper()
        self.STACKING_OVERRIDE_METHOD = self.STACKING_OVERRIDE_METHOD.lower()
        self.TP_MODE = self.TP_MODE.lower()
        self.ADJUSTED_DELTA_FALLBACK_MODE = self.ADJUSTED_DELTA_FALLBACK_MODE.lower()
        self.ADJUSTED_DIVERGENCE_ACTION = self.ADJUSTED_DIVERGENCE_ACTION.lower()
        self.CALIBRATION_METHOD = self.CALIBRATION_METHOD.lower()
        self.TRADE_STRATEGY = self.TRADE_STRATEGY.lower()

        if "STACKING_META_DAYS" not in self.model_fields_set:
            self.STACKING_META_DAYS = max(45, self.BOTTOM_TRAIN_DAYS)
        if "RETRAIN_DAYS" not in self.model_fields_set:
            self.RETRAIN_DAYS = max(14, self.BOTTOM_TRAIN_DAYS // 2)
        if self.ENTRY_WINRATE_TARGET_DAY is None:
            self.ENTRY_WINRATE_TARGET_DAY = self.ENTRY_WINRATE_TARGET
        if self.ENTRY_WINRATE_TARGET_NIGHT is None:
            self.ENTRY_WINRATE_TARGET_NIGHT = self.ENTRY_WINRATE_TARGET
        if self.ENTRY_META_ONBOARD_SAMPLES is None:
            self.ENTRY_META_ONBOARD_SAMPLES = max(self.ENTRY_WINRATE_MIN_SAMPLES * 2, 20)

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
