import math
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any


@dataclass
class NowcastResult:
    symbol: str
    interval: str
    timestamp: str
    price: float
    bottom_score: float
    components: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sigmoid(x: float) -> float:
    try:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)
    except Exception:
        return 0.5


class RealTimePredictor:
    """Lightweight heuristic predictor using latest candle features.

    NOTE:
    - This does NOT use the synthetic training artifacts under training/models.
    - It computes a plausible "bottom likelihood" score from common TA features
      that are already being computed in FeatureCalculator and stored in Candle.
    - You can later swap this with a proper trained model once a real dataset
      aligned to Candle schema is available.
    """

    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval
        self.last_results: Dict[str, NowcastResult] = {}

    def predict_from_row(self, row: Any) -> NowcastResult:
        # Extract inputs with safe defaults
        close = float(getattr(row, 'close', 0.0) or 0.0)
        rsi = float(getattr(row, 'rsi_14', 50.0) or 50.0)
        bb_pct_b = float(getattr(row, 'bb_pct_b_20_2', 0.5) or 0.5)
        dd20 = float(getattr(row, 'drawdown_from_max_20', 0.0) or 0.0)
        macd_hist = float(getattr(row, 'macd_hist', 0.0) or 0.0)
        volz = float(getattr(row, 'vol_z_20', 0.0) or 0.0)
        willr = float(getattr(row, 'williams_r_14', -50.0) or -50.0)

        # Feature transforms into [0,1] encouraging "bottom" signal when near extremes
        # Lower RSI -> higher score (below 30 strong), clamp to [0,1]
        s_rsi = max(0.0, min(1.0, (30.0 - rsi) / 30.0))
        # Bollinger %B near 0 -> higher score
        s_bb = max(0.0, min(1.0, (0.25 - bb_pct_b) / 0.25))
        # Drawdown (negative) magnitude up to ~-5% maps toward 1
        s_dd = max(0.0, min(1.0, -dd20 / 0.05))
        # MACD histogram negative (below 0) favors bottom; invert sign and scale
        s_macd = max(0.0, min(1.0, -macd_hist / (abs(macd_hist) + 1e-6))) * 0.5
        # Volume z-score > 1 may indicate capitulation; map (volz-1)/3 to [0,1]
        s_vol = max(0.0, min(1.0, (volz - 1.0) / 3.0))
        # Williams %R near -100 -> bottom; map (-80 to -100) into 0..1
        s_wr = max(0.0, min(1.0, (-80.0 - willr) / 20.0))

        # Weighted combination -> logit -> sigmoid to get [0,1]
        # Weights favor RSI, %B, and drawdown; volume helps when extreme.
        w_rsi = 1.8
        w_bb = 1.5
        w_dd = 1.2
        w_macd = 0.6
        w_vol = 0.8
        w_wr = 0.9
        logit = (
            w_rsi * s_rsi
            + w_bb * s_bb
            + w_dd * s_dd
            + w_macd * s_macd
            + w_vol * s_vol
            + w_wr * s_wr
            - 2.0  # bias
        )
        score = float(_sigmoid(logit))

        comp = {
            'rsi_14': rsi,
            'bb_pct_b_20_2': bb_pct_b,
            'drawdown_from_max_20': dd20,
            'macd_hist': macd_hist,
            'vol_z_20': volz,
            'williams_r_14': willr,
            's_rsi': s_rsi,
            's_bb': s_bb,
            's_dd': s_dd,
            's_macd': s_macd,
            's_vol': s_vol,
            's_wr': s_wr,
            'logit': logit,
        }

        ts = getattr(row, 'close_time', None) or getattr(row, 'open_time', None)
        ts_str = ts.isoformat() if isinstance(ts, datetime) else datetime.utcnow().isoformat()

        return NowcastResult(
            symbol=getattr(row, 'symbol', self.symbol),
            interval=getattr(row, 'interval', self.interval),
            timestamp=ts_str,
            price=close,
            bottom_score=score,
            components=comp,
        )
