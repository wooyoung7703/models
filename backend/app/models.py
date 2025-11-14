from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field, Index


class Candle(SQLModel, table=True):
    __tablename__ = "candles"

    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    exchange_type: str = Field(index=True)  # spot or futures
    interval: str = Field(index=True, default="1m")

    # OHLCV
    open_time: datetime = Field(index=True)
    close_time: datetime = Field(index=True)
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: Optional[int] = 0

    # Precomputed features (examples)
    ret_1: Optional[float] = None
    ret_5: Optional[float] = None
    ret_15: Optional[float] = None

    ma_5: Optional[float] = None
    ma_20: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None

    rsi_14: Optional[float] = None
    vol_20: Optional[float] = None  # rolling std of returns
    atr_14: Optional[float] = None  # average true range

    body: Optional[float] = None  # |close - open|
    upper_shadow: Optional[float] = None  # high - max(open, close)
    lower_shadow: Optional[float] = None  # min(open, close) - low

    # New advanced features for bottom detection
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None

    bb_upper_20_2: Optional[float] = None
    bb_lower_20_2: Optional[float] = None
    bb_pct_b_20_2: Optional[float] = None
    bb_bandwidth_20_2: Optional[float] = None

    stoch_k_14_3: Optional[float] = None
    stoch_d_14_3: Optional[float] = None
    williams_r_14: Optional[float] = None

    mfi_14: Optional[float] = None
    obv: Optional[float] = None
    cmf_20: Optional[float] = None
    vol_z_20: Optional[float] = None

    dist_min_close_20: Optional[float] = None
    dist_min_close_50: Optional[float] = None
    drawdown_from_max_20: Optional[float] = None

    di_plus_14: Optional[float] = None
    di_minus_14: Optional[float] = None
    adx_14: Optional[float] = None

    run_up: Optional[int] = None
    run_down: Optional[int] = None
    body_pct_of_range: Optional[float] = None
    range_hl: Optional[float] = None

    # Additional features to reach 60 total (including OHLCV)
    ma_50: Optional[float] = None
    ma_200: Optional[float] = None
    ema_9: Optional[float] = None
    ema_50: Optional[float] = None
    rsi_7: Optional[float] = None
    rsi_21: Optional[float] = None
    roc_10: Optional[float] = None
    cci_20: Optional[float] = None
    atr_7: Optional[float] = None
    vol_50: Optional[float] = None
    bb_pct_b_50_2: Optional[float] = None
    dist_min_close_100: Optional[float] = None
    price_to_ma_20: Optional[float] = None
    vwap_20_dev: Optional[float] = None
    vwap_50_dev: Optional[float] = None
    zscore_close_20: Optional[float] = None
    bb_bandwidth_50_2: Optional[float] = None
    price_to_ma_50: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

    __table_args__ = (
        Index(
            "ix_unique_candle",
            "symbol",
            "exchange_type",
            "interval",
            "open_time",
            unique=True,
        ),
    )


class FeatureState(SQLModel, table=True):
    __tablename__ = "feature_state"

    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    exchange_type: str = Field(index=True)
    interval: str = Field(index=True, default="1m")
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    version: int = Field(default=1)
    state_json: str  # JSON serialized internal state

    __table_args__ = (
        Index(
            "ix_unique_feature_state",
            "symbol",
            "exchange_type",
            "interval",
            unique=True,
        ),
    )


class Trade(SQLModel, table=True):
    __tablename__ = "trades"

    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    exchange_type: str = Field(index=True, default="futures")
    interval: str = Field(index=True, default="1m")
    side: str = Field(default="long", index=True)
    leverage: int = Field(default=10)
    status: str = Field(default="open", index=True)  # open | closed

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    closed_at: Optional[datetime] = Field(default=None, index=True)

    entry_price: float
    avg_price: float
    quantity: float = Field(default=0.0)  # total base qty

    max_adds: int = Field(default=1000)  # effectively unlimited adds for stress test
    adds_done: int = Field(default=0)

    take_profit_pct: float = Field(default=0.01)   # +1%
    stop_loss_pct: float = Field(default=-0.005)   # -0.5%

    # Trailing TP settings (optional). If tp_mode == 'trailing', use trigger/step/giveback.
    # Trailing TP runtime state is managed in TradeManager (not persisted in DB)

    last_price: Optional[float] = None
    pnl_pct_snapshot: Optional[float] = None
    strategy_json: Optional[str] = None  # store entry condition snapshot


class TradeFill(SQLModel, table=True):
    __tablename__ = "trade_fills"

    id: Optional[int] = Field(default=None, primary_key=True)
    trade_id: int = Field(index=True, foreign_key="trades.id")
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    price: float
    quantity: float = Field(default=0.0)

    symbol: str = Field(index=True)
    exchange_type: str = Field(index=True, default="futures")
    interval: str = Field(index=True, default="1m")
