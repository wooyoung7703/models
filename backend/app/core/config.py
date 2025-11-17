import os
from pathlib import Path
from dotenv import load_dotenv

# Load both project root .env (if present) then backend/.env so backend can override.
_CONFIG_FILE = Path(__file__).resolve()
_BACKEND_DIR = _CONFIG_FILE.parents[2]  # .../backend
_ROOT_DIR = _CONFIG_FILE.parents[3]     # repo root
root_env = _ROOT_DIR / '.env'
backend_env = _BACKEND_DIR / '.env'
if root_env.exists():
    load_dotenv(dotenv_path=root_env, override=False)
if backend_env.exists():
    # Do NOT override real environment variables with backend/.env.
    # This ensures values provided by the running process (e.g., CI, shell) take precedence.
    load_dotenv(dotenv_path=backend_env, override=False)


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
    # Websocket connection open timeout (seconds) to distinguish slow DNS vs hard failure
    WS_OPEN_TIMEOUT_SECONDS: int = int(os.getenv("WS_OPEN_TIMEOUT_SECONDS", "15"))

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
    PREDICT_INTERVAL_SECONDS: int = int(os.getenv("PREDICT_INTERVAL_SECONDS", "10"))
    # Resampler tick interval seconds (build 5m/15m aggregates)
    RESAMPLER_INTERVAL_SECONDS: int = int(os.getenv("RESAMPLER_INTERVAL_SECONDS", "55"))
    # Feature snapshot cadence (every N closed candles per symbol)
    FEATURE_SNAPSHOT_EVERY: int = int(os.getenv("FEATURE_SNAPSHOT_EVERY", "50"))

    # Shared training window for bottom-detection models (days of history)
    BOTTOM_TRAIN_DAYS: int = int(os.getenv("BOTTOM_TRAIN_DAYS", "14"))

    # Alerts & error throttling
    ALERT_WEBHOOK_URL: str | None = os.getenv("ALERT_WEBHOOK_URL")
    ALERT_COOLDOWN_SECONDS: int = int(os.getenv("ALERT_COOLDOWN_SECONDS", "300"))
    ERROR_LOG_MIN_INTERVAL_SECONDS: int = int(os.getenv("ERROR_LOG_MIN_INTERVAL_SECONDS", "60"))

    # --- Model integration settings (multi-model & stacking) ---
    # Primary model selection (heuristic, xgb, lstm, tf, stacking)
    MODEL_TYPE: str = os.getenv("MODEL_TYPE", "heuristic")
    # Whether to compute base model probabilities (if models available)
    ENABLE_BASE_MODELS: bool = True
    # Whether to apply stacking meta ensemble
    ENABLE_STACKING: bool = True
    # Paths to trained artifacts (override in .env if relocated)
    _BASE_MODELS_DIR = Path(__file__).resolve().parents[2] / "app" / "training" / "models"
    MODEL_XGB_PATH: str = os.getenv("MODEL_XGB_PATH", str(_BASE_MODELS_DIR / "xgb_bottom_real_1m_30d.pkl"))
    MODEL_LSTM_PATH: str = os.getenv("MODEL_LSTM_PATH", str(_BASE_MODELS_DIR / "lstm_model.pt"))
    MODEL_TRANSFORMER_PATH: str = os.getenv("MODEL_TRANSFORMER_PATH", str(_BASE_MODELS_DIR / "transformer_model.pt"))
    STACKING_META_PATH: str = os.getenv("STACKING_META_PATH", str(_BASE_MODELS_DIR / "stacking_meta.json"))
    # Optional threshold override for stacking decision (float)
    # Default static threshold tuned via 90d backtest; override with env or adaptive if desired
    STACKING_THRESHOLD: float = float(os.getenv("STACKING_THRESHOLD", "0.78"))  # set -1 to prefer sidecar/adaptive
    # Optional override for ensemble method at runtime (rollback/risk control): '', 'logistic','dynamic','bayes','mean'
    STACKING_OVERRIDE_METHOD: str = os.getenv("STACKING_OVERRIDE_METHOD", "").lower()
    # If true, stacking will include all ready base models present at runtime.
    # When meta lacks weights for extras, combiner safely falls back to 'mean'.
    STACKING_INCLUDE_ALL_READY: bool = os.getenv("STACKING_INCLUDE_ALL_READY", "1") in {"1","true","True"}
    # Sequence length for LSTM/Transformer if enabled
    SEQ_LEN: int = int(os.getenv("SEQ_LEN", "30"))
    # Minimum recent sequences required before seq model inference
    SEQ_MIN_READY: int = int(os.getenv("SEQ_MIN_READY", "10"))
    # Trading: cooldown between DCA adds (seconds)
    ADD_COOLDOWN_SECONDS: int = int(os.getenv("ADD_COOLDOWN_SECONDS", "600"))
    # Trading: default TP/SL and DCA limits
    # Compromise live mode: target +0.8% net move
    TAKE_PROFIT_PCT: float = float(os.getenv("TAKE_PROFIT_PCT", "0.008"))   # +0.8% compromise TP
    STOP_LOSS_PCT_RAW: str | None = os.getenv("STOP_LOSS_PCT", None)
    @property
    def STOP_LOSS_PCT(self) -> float | None:
        try:
            return float(self.STOP_LOSS_PCT_RAW) if self.STOP_LOSS_PCT_RAW is not None else None
        except Exception:
            return None
    # Trailing TP defaults
    # Fixed TP for compromise policy (was 'trailing')
    TP_MODE: str = os.getenv("TP_MODE", "fixed").lower()  # 'fixed' | 'trailing'
    TP_TRIGGER: float = float(os.getenv("TP_TRIGGER", "0.005"))
    TP_STEP: float = float(os.getenv("TP_STEP", "0.0005"))
    TP_GIVEBACK: float = float(os.getenv("TP_GIVEBACK", "0.001"))

    # --- Fees (applied in backtests and PnL snapshots at close) ---
    # Percent fees (e.g., 0.0004 = 0.04%) per side; defaults approximate futures taker fees
    FEE_ENTRY_PCT: float = float(os.getenv("FEE_ENTRY_PCT", "0.0004"))
    FEE_EXIT_PCT: float = float(os.getenv("FEE_EXIT_PCT", "0.0004"))

    # --- Backtest/Exit policy toggles ---
    # If true, ignore stop loss checks (no SL)
    # Disable SL by default for compromise policy (evaluate adds + controlled TP)
    DISABLE_STOP_LOSS: bool = os.getenv("DISABLE_STOP_LOSS", "1") in {"1","true","True"}
    # If true, evaluate fixed TP using net-of-fees PnL instead of gross move
    # Evaluate TP on net-of-fees move to ensure real profit capture
    TP_DECISION_ON_NET: bool = os.getenv("TP_DECISION_ON_NET", "1") in {"1","true","True"}

    # --- Calibration & Smoothing & Adaptive Threshold ---
    # Enable application of calibration parameters from stacking meta (if present)
    ENABLE_CALIBRATION: bool = os.getenv("ENABLE_CALIBRATION", "1") in {"1", "true", "True"}
    # Preferred calibration method fallback if meta lacks (platt|isotonic|none)
    CALIBRATION_METHOD: str = os.getenv("CALIBRATION_METHOD", "platt")
    # Probability smoothing via exponential moving average on stacking probability
    ENABLE_PROB_SMOOTHING: bool = os.getenv("ENABLE_PROB_SMOOTHING", "1") in {"1", "true", "True"}
    PROB_EMA_ALPHA: float = float(os.getenv("PROB_EMA_ALPHA", "0.3"))  # higher -> more reactive
    # Adaptive thresholding: replace static threshold with rolling quantile of recent probabilities
    ENABLE_ADAPTIVE_THRESHOLD: bool = os.getenv("ENABLE_ADAPTIVE_THRESHOLD", "1") in {"1", "true", "True"}
    ADAPTIVE_THRESHOLD_QUANTILE: float = float(os.getenv("ADAPTIVE_THRESHOLD_QUANTILE", "0.98"))  # e.g. 0.98 upper tail
    ADAPTIVE_MIN_HISTORY: int = int(os.getenv("ADAPTIVE_MIN_HISTORY", "120"))  # minimum samples before adapt kicks in
    ADAPTIVE_HISTORY_MAX: int = int(os.getenv("ADAPTIVE_HISTORY_MAX", "2000"))  # cap history size (FIFO)
    ADAPTIVE_THRESHOLD_CLAMP_LOW: float = float(os.getenv("ADAPTIVE_THRESHOLD_CLAMP_LOW", "0.50"))
    ADAPTIVE_THRESHOLD_CLAMP_HIGH: float = float(os.getenv("ADAPTIVE_THRESHOLD_CLAMP_HIGH", "0.995"))

    # --- Label schema versioning ---
    # Tag the bottom-label definition to keep artifacts compatible across changes
    LABEL_SCHEMA_VERSION: str = os.getenv("LABEL_SCHEMA_VERSION", "bottom_v1")

    # Adaptive volatility labeling (scale tolerance by recent ATR%)
    ENABLE_VOL_LABELING: bool = os.getenv("ENABLE_VOL_LABELING", "0") in {"1","true","True"}
    VOL_LABEL_ATR_FEATURE: str = os.getenv("VOL_LABEL_ATR_FEATURE", "atr_14")
    VOL_LABEL_BASE_ATR_PCT: float = float(os.getenv("VOL_LABEL_BASE_ATR_PCT", "0.01"))  # e.g., 1% baseline
    VOL_LABEL_MIN_SCALE: float = float(os.getenv("VOL_LABEL_MIN_SCALE", "0.5"))
    VOL_LABEL_MAX_SCALE: float = float(os.getenv("VOL_LABEL_MAX_SCALE", "2.0"))

    # --- AutoEncoder feature augmentation ---
    AE_AUGMENT: bool = os.getenv("AE_AUGMENT", "0") in {"1","true","True"}
    AE_MODEL_PATH: str = os.getenv("AE_MODEL_PATH", str(Path(__file__).resolve().parents[2] / "app" / "training" / "models" / "ae_timestep.pt"))

    # --- Probability drift monitoring (stacking probability distribution shift) ---
    PROB_DRIFT_ENABLED: bool = os.getenv("PROB_DRIFT_ENABLED", "0") in {"1","true","True"}
    PROB_DRIFT_RECENT_SIZE: int = int(os.getenv("PROB_DRIFT_RECENT_SIZE", "500"))  # recent window sample size
    PROB_DRIFT_MIN_BASELINE: int = int(os.getenv("PROB_DRIFT_MIN_BASELINE", "50"))  # minimum baseline samples required
    PROB_DRIFT_KS_P_THRESHOLD: float = float(os.getenv("PROB_DRIFT_KS_P_THRESHOLD", "0.01"))  # p-value below => drift
    PROB_DRIFT_WASSERSTEIN_THRESHOLD: float = float(os.getenv("PROB_DRIFT_WASSERSTEIN_THRESHOLD", "0.15"))  # avg quantile distance
    PROB_DRIFT_MIN_INTERVAL_HOURS: float = float(os.getenv("PROB_DRIFT_MIN_INTERVAL_HOURS", "12"))  # cooldown between auto drift retrains
    PROB_DRIFT_CONSECUTIVE_REQUIRED: int = int(os.getenv("PROB_DRIFT_CONSECUTIVE_REQUIRED", "2"))  # require N consecutive drift flags before retrain

    # --- Event-driven retraining (trade close) ---
    # Enable automatic retraining sequence immediately after any trade closes (TP or SL)
    RETRAIN_ON_TRADE_CLOSE: bool = os.getenv("RETRAIN_ON_TRADE_CLOSE", "0") in {"1", "true", "True"}
    # Minimum interval between automated trade-close retrains (hours)
    RETRAIN_MIN_INTERVAL_HOURS: float = float(os.getenv("RETRAIN_MIN_INTERVAL_HOURS", "12"))
    # Alternate day windows for different retrain types
    # Monthly/base long horizon (already BOTTOM_TRAIN_DAYS)
    STACKING_META_DAYS: int = int(os.getenv("STACKING_META_DAYS", str(BOTTOM_TRAIN_DAYS if BOTTOM_TRAIN_DAYS >= 45 else 45)))
    # Event-driven (trade-close) shorter horizon for rapid adaptation
    RETRAIN_DAYS: int = int(os.getenv("RETRAIN_DAYS", str(max(14, BOTTOM_TRAIN_DAYS // 2))))
    # Optional separate stacking days for event retrain; fallback to STACKING_META_DAYS
    RETRAIN_STACKING_DAYS_RAW: str | None = os.getenv("RETRAIN_STACKING_DAYS")
    @property
    def RETRAIN_STACKING_DAYS(self) -> int:
        try:
            if self.RETRAIN_STACKING_DAYS_RAW is not None and self.RETRAIN_STACKING_DAYS_RAW.strip():
                return int(self.RETRAIN_STACKING_DAYS_RAW)
        except Exception:
            pass
        return self.STACKING_META_DAYS

    # --- Model file watcher (backend auto-reload) ---
    MODEL_WATCH_ENABLED: bool = os.getenv("MODEL_WATCH_ENABLED", "1") in {"1","true","True"}
    MODEL_WATCH_INTERVAL_SECONDS: int = int(os.getenv("MODEL_WATCH_INTERVAL_SECONDS", "60"))

    # --- Bottom vs Forecast meta (optional runtime probability adjustment) ---
    # Path to logistic meta JSON produced by scripts/bottom_vs_forecast_analysis.py
    # Default points to repo root data/ directory.
    BOTTOM_VS_FORECAST_META_PATH: str = os.getenv(
        "BOTTOM_VS_FORECAST_META_PATH",
        str(Path(__file__).resolve().parents[3] / "data" / "bottom_vs_forecast_meta.json"),
    )
    # Real-time naive forecast horizon (steps) for bottom-vs-forecast adjustment
    FORECAST_HORIZON: int = int(os.getenv("FORECAST_HORIZON", "12"))
    # ATR smoothing alpha for forward path volatility estimate
    FORECAST_ATR_ALPHA: float = float(os.getenv("FORECAST_ATR_ALPHA", "0.2"))
    # Clamp for adjusted probability delta vs base (absolute). If <=0 disable.
    ADJUSTED_DELTA_CLAMP: float = float(os.getenv("ADJUSTED_DELTA_CLAMP", "0.25"))
    # Fallback mode when |delta| > clamp: 'limit' -> clip; 'revert' -> keep base
    ADJUSTED_DELTA_FALLBACK_MODE: str = os.getenv("ADJUSTED_DELTA_FALLBACK_MODE", "limit").lower()
    # Divergence watchdog thresholds (override adjusted prob if unstable)
    ADJUSTED_DIVERGENCE_MEAN_THRESHOLD: float = float(os.getenv("ADJUSTED_DIVERGENCE_MEAN_THRESHOLD", "0.15"))
    ADJUSTED_DIVERGENCE_ABS_MEAN_THRESHOLD: float = float(os.getenv("ADJUSTED_DIVERGENCE_ABS_MEAN_THRESHOLD", "0.18"))
    ADJUSTED_DIVERGENCE_MIN_SAMPLES: int = int(os.getenv("ADJUSTED_DIVERGENCE_MIN_SAMPLES", "60"))
    ADJUSTED_DIVERGENCE_ACTION: str = os.getenv("ADJUSTED_DIVERGENCE_ACTION", "revert").lower()  # 'revert' | 'log'

    # --- Automated meta retraining (bottom-vs-forecast logistic layer) ---
    META_RETRAIN_ENABLED: bool = os.getenv("META_RETRAIN_ENABLED", "0") in {"1","true","True"}
    # Either run every N minutes (takes precedence if >0) or at daily HH:MM.
    META_RETRAIN_EVERY_MINUTES: int = int(os.getenv("META_RETRAIN_EVERY_MINUTES", "0"))  # e.g. 720 for 12h
    META_RETRAIN_DAILY_AT: str = os.getenv("META_RETRAIN_DAILY_AT", "03:15")  # HH:MM local scheduler TZ
    META_RETRAIN_MIN_INTERVAL_MINUTES: int = int(os.getenv("META_RETRAIN_MIN_INTERVAL_MINUTES", "60"))
    # Relative Brier improvement requirement to overwrite meta (passed to script)
    META_RETRAIN_MIN_REL_BRIER_IMPROVE: float = float(os.getenv("META_RETRAIN_MIN_REL_BRIER_IMPROVE", "0.005"))
    # Evaluation CSV path (new labeled rows + forward preds). Default sample if not provided.
    BOTTOM_VS_FORECAST_EVAL_CSV_PATH: str = os.getenv(
        "BOTTOM_VS_FORECAST_EVAL_CSV_PATH",
        str(Path(__file__).resolve().parents[3] / "data" / "bottom_eval_sample.csv"),
    )
    # Meta JSON output (reuse existing path for overwrite)
    BOTTOM_VS_FORECAST_META_OUT_PATH: str = os.getenv(
        "BOTTOM_VS_FORECAST_META_OUT_PATH",
        BOTTOM_VS_FORECAST_META_PATH,
    )

    # --- Entry Meta (precision-focused gating for win-rate) ---
    # Enable secondary logistic meta to gate entries with high precision.
    ENABLE_ENTRY_META: bool = os.getenv("ENABLE_ENTRY_META", "0") in {"1","true","True"}
    # Path to entry meta JSON (format: {"features": [...], "coef": [...], "intercept": ..., "threshold": 0.xx (optional)})
    ENTRY_META_PATH: str = os.getenv(
        "ENTRY_META_PATH",
        str(Path(__file__).resolve().parents[3] / "data" / "entry_meta.json"),
    )
    # If JSON lacks threshold, use this env threshold (higher favors precision)
    ENTRY_META_THRESHOLD: float = float(os.getenv("ENTRY_META_THRESHOLD", "0.90"))
    # Whether to use adjusted prob (after BVF) as the primary feature
    ENTRY_META_USE_ADJUSTED_PROB: bool = os.getenv("ENTRY_META_USE_ADJUSTED_PROB", "1") in {"1","true","True"}
    # Optional cooldown to avoid excessive recomputation (seconds, 0=disabled)
    ENTRY_META_MIN_INTERVAL_SECONDS: int = int(os.getenv("ENTRY_META_MIN_INTERVAL_SECONDS", "0"))
    # Require entry_meta decision true for trade entry/add (improves precision). If disabled, fallback to stacking decision only.
    ENTRY_META_GATE_ENABLED: bool = os.getenv("ENTRY_META_GATE_ENABLED", "1") in {"1","true","True"}
    # Rolling win-rate window for dynamic entry threshold
    ENTRY_WINRATE_WINDOW: int = int(os.getenv("ENTRY_WINRATE_WINDOW", "30"))
    ENTRY_WINRATE_MIN_SAMPLES: int = int(os.getenv("ENTRY_WINRATE_MIN_SAMPLES", "10"))
    ENTRY_WINRATE_TARGET: float = float(os.getenv("ENTRY_WINRATE_TARGET", "0.75"))
    # Optional session-based targets (day/night). Enable split to use.
    ENTRY_SESSION_SPLIT_ENABLED: bool = os.getenv("ENTRY_SESSION_SPLIT_ENABLED", "0") in {"1","true","True"}
    ENTRY_SESSION_DAY_START_HOUR: int = int(os.getenv("ENTRY_SESSION_DAY_START_HOUR", "9"))
    ENTRY_SESSION_DAY_END_HOUR: int = int(os.getenv("ENTRY_SESSION_DAY_END_HOUR", "21"))
    ENTRY_WINRATE_TARGET_DAY: float = float(os.getenv("ENTRY_WINRATE_TARGET_DAY", str(ENTRY_WINRATE_TARGET)))
    ENTRY_WINRATE_TARGET_NIGHT: float = float(os.getenv("ENTRY_WINRATE_TARGET_NIGHT", str(ENTRY_WINRATE_TARGET)))
    # Dynamic adjustment step and clamps
    ENTRY_META_DYNAMIC_STEP: float = float(os.getenv("ENTRY_META_DYNAMIC_STEP", "0.01"))
    ENTRY_META_DYNAMIC_MIN: float = float(os.getenv("ENTRY_META_DYNAMIC_MIN", "0.85"))
    ENTRY_META_DYNAMIC_MAX: float = float(os.getenv("ENTRY_META_DYNAMIC_MAX", "0.98"))
    # Onboarding phase: soften per-symbol adaptation when samples are small
    ENTRY_META_ONBOARD_SAMPLES: int = int(os.getenv("ENTRY_META_ONBOARD_SAMPLES", str(max(ENTRY_WINRATE_MIN_SAMPLES * 2, 20))))
    ENTRY_META_ONBOARD_STEP_SCALE: float = float(os.getenv("ENTRY_META_ONBOARD_STEP_SCALE", "0.5"))
    # Persistence for dynamic thresholds (global and per-symbol)
    ENTRY_META_STATE_PATH: str = os.getenv(
        "ENTRY_META_STATE_PATH",
        str(Path(__file__).resolve().parents[2] / "data" / "entry_meta_state.json"),
    )



settings = Settings()
