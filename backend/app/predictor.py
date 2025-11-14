import math
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional
from .model_adapters import registry


@dataclass
class NowcastResult:
    symbol: str
    interval: str
    timestamp: str
    price: float
    bottom_score: float
    price_source: str  # 'live' or 'closed'
    components: Dict[str, float]
    # Optional extended outputs
    base_probs: Optional[Dict[str, Optional[float]]] = None
    stacking: Optional[Dict[str, Any]] = None
    # Optional per-model debug metadata (e.g., seq_len, feature_dim, padded)
    base_info: Optional[Dict[str, Any]] = None

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
        # Smoothing / adaptive threshold state
        self._ema_prob = None
        self._prob_history = []

    def predict_from_row(self, row: Any, price_override: Optional[float] = None, price_source: Optional[str] = None) -> NowcastResult:
        # Extract inputs with safe defaults
        close = float(getattr(row, 'close', 0.0) or 0.0)
        if price_override is not None:
            close = float(price_override)
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

        comp: Dict[str, Any] = {
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
        # price_source captured separately in NowcastResult (avoid polluting components)

        ts = getattr(row, 'close_time', None) or getattr(row, 'open_time', None)
        ts_str = ts.isoformat() if isinstance(ts, datetime) else datetime.utcnow().isoformat()

        # Collect base model predictions (soft fail if adapters not ready)
        base_probs: Dict[str, Optional[float]] = {}
        base_logits: Dict[str, Optional[float]] = {}
        base_info: Dict[str, Any] = {}
        stacking: Optional[Dict[str, Any]] = None
        try:
            if not registry.adapters and not registry.stacking:
                registry.load_from_settings()
            for name, adapter in registry.adapters.items():
                if not adapter.ready():
                    base_probs[name] = None
                    base_logits[name] = None
                    base_info[name] = {"ready": False}
                    continue
                pres = adapter.predict(row, live_price=price_override)
                base_probs[name] = pres.prob if pres.ready else None
                # collect lightweight debug info from adapter's PredictResult
                try:
                    base_info[name] = {"ready": pres.ready, **(pres.details or {})}
                except Exception:
                    base_info[name] = {"ready": pres.ready}
                # derive per-model logit when probability present for observability
                try:
                    if pres.ready and pres.prob is not None:
                        p = float(pres.prob)
                        eps = 1e-9
                        p = min(max(p, eps), 1 - eps)
                        base_logits[name] = math.log(p / (1 - p))
                    else:
                        base_logits[name] = None
                except Exception:
                    base_logits[name] = None
            if registry.stacking and registry.stacking.ready():
                # Simple real-time regime detection using volume z-score and drawdown
                regime = None
                try:
                    if volz is not None:
                        if float(volz) >= 1.0:
                            regime = 'high_vol'
                        else:
                            regime = 'low_vol'
                except Exception:
                    regime = None
                stacking = registry.stacking.combine(base_probs, regime=regime)
        except Exception as e:
            stacking = {"ready": False, "error": str(e)}

        # Augment stacking block with derived fields for frontend convenience
        if stacking and stacking.get('ready'):
            try:
                sp = float(stacking.get('prob', 0.0))
                raw_sp = float(stacking.get('raw_prob', sp))
                th = stacking.get('threshold')
                # --- Probability smoothing (EMA) ---
                from .core.config import settings as _s
                if _s.ENABLE_PROB_SMOOTHING:
                    alpha = max(0.0, min(1.0, _s.PROB_EMA_ALPHA))
                    if self._ema_prob is None:
                        self._ema_prob = sp
                    else:
                        self._ema_prob = self._ema_prob * (1 - alpha) + sp * alpha
                    stacking['prob_smoothed'] = float(self._ema_prob)
                # Maintain history for adaptive threshold
                try:
                    self._prob_history.append(sp)
                    max_hist = int(_s.ADAPTIVE_HISTORY_MAX)
                    if len(self._prob_history) > max_hist:
                        self._prob_history = self._prob_history[-max_hist:]
                except Exception:
                    pass
                # --- Adaptive threshold ---
                adaptive_th = None
                if _s.ENABLE_ADAPTIVE_THRESHOLD and len(self._prob_history) >= _s.ADAPTIVE_MIN_HISTORY:
                    import numpy as _np
                    q = min(0.999, max(0.5, _s.ADAPTIVE_THRESHOLD_QUANTILE))
                    adaptive_th = float(_np.quantile(self._prob_history, q))
                    adaptive_th = max(_s.ADAPTIVE_THRESHOLD_CLAMP_LOW, min(_s.ADAPTIVE_THRESHOLD_CLAMP_HIGH, adaptive_th))
                    stacking['threshold_adaptive'] = adaptive_th
                    stacking['threshold_source_adaptive'] = 'quantile'
                # Choose final threshold precedence: explicit/env > adaptive > sidecar
                final_th = th
                source = stacking.get('threshold_source')
                if final_th is None and adaptive_th is not None:
                    final_th = adaptive_th
                    source = 'adaptive'
                elif final_th is not None and (source in {'sidecar', None}) and adaptive_th is not None:
                    # if sidecar and adaptive available, favor adaptive for dynamic regime
                    final_th = adaptive_th
                    source = 'adaptive'
                # Decision uses calibrated + smoothed prob if available
                p_for_decision_f = float(sp)
                if 'prob_smoothed' in stacking:
                    val_ps = stacking.get('prob_smoothed')
                    if isinstance(val_ps, (int, float)):
                        p_for_decision_f = float(val_ps)
                stacking['threshold'] = final_th
                stacking['threshold_source_final'] = source
                if isinstance(final_th, (int, float)):
                    final_th_val = float(final_th)
                    decision = p_for_decision_f >= final_th_val
                    stacking['decision'] = decision
                    stacking['above_threshold'] = decision
                    diff = p_for_decision_f - final_th_val
                    stacking['margin'] = diff
                    stacking['confidence'] = abs(diff)
                else:
                    stacking['decision'] = None
                    stacking['above_threshold'] = None
                stacking['prob_raw'] = raw_sp
                # Echo regime in stacking block if provided by combiner
                if stacking.get('regime_used') is None:
                    # if combiner didn't use regime weights, still expose sensed regime
                    try:
                        stacking['regime_sensed'] = 'high_vol' if float(volz) >= 1.0 else 'low_vol'
                    except Exception:
                        pass
                stacking['prob_final'] = p_for_decision_f
            except Exception:
                pass

        return NowcastResult(
            symbol=getattr(row, 'symbol', self.symbol),
            interval=getattr(row, 'interval', self.interval),
            timestamp=ts_str,
            price=close,
            bottom_score=score,
            price_source=price_source or 'closed',
            components=comp,
            base_probs=base_probs or None,
            stacking=stacking,
            base_info={**(base_info or {}), "base_logits": base_logits or None} if base_info else {"base_logits": base_logits or None},
        )
