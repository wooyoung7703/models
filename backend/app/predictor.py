import math
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional

from .heuristic_scoring import compute_bottom_score
from .model_adapters import registry

log = logging.getLogger(__name__)


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
        # Price history for naive forecast generation
        from collections import deque
        self._price_history = deque(maxlen=500)  # keep recent closes
        # Adjusted probability drift monitoring
        self._adjust_delta_history = deque(maxlen=500)

    def predict_from_row(self, row: Any, price_override: Optional[float] = None, price_source: Optional[str] = None) -> NowcastResult:
        # Extract inputs with safe defaults
        close = float(getattr(row, 'close', 0.0) or 0.0)
        if price_override is not None:
            close = float(price_override)
        heuristic = compute_bottom_score(row, price_override=price_override)
        score = heuristic.prob
        comp = dict(heuristic.components)
        # For backward compatibility keep component payload identical (exclude close)
        comp.pop('close', None)
        volz = comp.get('vol_z_20', 0.0)
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
                # --- Forecast-based adjustment (enhanced with ATR & delta clamp) ---
                try:
                    from .core.config import settings as _s
                    horizon = int(getattr(_s, 'FORECAST_HORIZON', 12))
                    # Update price history
                    self._price_history.append(close)
                    prices_list = list(self._price_history)
                    if len(prices_list) >= 10 and registry.stacking and getattr(registry.stacking, 'bvf_coef', None):
                        # Simple drift estimation: mean of last diffs
                        import numpy as _np
                        window = prices_list[-50:] if len(prices_list) >= 50 else prices_list
                        diffs = _np.diff(window) if len(window) >= 2 else _np.array([0.0])
                        drift = float(_np.mean(diffs)) if diffs.size else 0.0
                        vol = float(_np.std(diffs)) if diffs.size else 0.0
                        # ATR estimate (avg absolute diff) with smoothing
                        atr_raw = float(_np.mean(_np.abs(diffs))) if diffs.size else 0.0
                        atr_alpha = max(0.0, min(1.0, getattr(_s, 'FORECAST_ATR_ALPHA', 0.2)))
                        prev_atr = getattr(self, '_atr_est', None)
                        if prev_atr is None:
                            self._atr_est = atr_raw
                        else:
                            self._atr_est = prev_atr * (1 - atr_alpha) + atr_raw * atr_alpha
                        atr_est = float(self._atr_est)
                        # Build central drift path and apply downward ATR envelope for min projection
                        fwd_prices = []
                        for k in range(1, horizon + 1):
                            # envelope uses sqrt(k) scaling for diffusion-like expansion
                            down_envelope = atr_est * (k ** 0.5)
                            fwd_prices.append(close + drift * k - 0.5 * down_envelope)
                        # Forecast features (align with analysis script semantics)
                        f_min = min(fwd_prices)
                        f_mean = float(_np.mean(fwd_prices))
                        room_to_min = (f_min - close) / close if close != 0 else 0.0
                        rel_to_mean = (close - f_mean) / f_mean if f_mean != 0 else 0.0
                        exp_ret = f_mean / close - 1.0 if close != 0 else 0.0
                        stacking['forecast_features'] = {
                            'room_to_forecast_min': room_to_min,
                            'rel_to_forecast_mean': rel_to_mean,
                            'forecast_expected_return': exp_ret,
                            'f_min': f_min,
                            'f_mean': f_mean,
                            'horizon': horizon,
                            'drift_estimate': drift,
                            'vol_estimate': vol,
                            'atr_estimate': atr_est,
                        }
                        coef = registry.stacking.bvf_coef
                        intercept = registry.stacking.bvf_intercept
                        if coef and intercept is not None:
                            # Feature vector order: bottom_prob (stack prob), room, rel, exp_ret
                            x_vec = [float(sp), room_to_min, rel_to_mean, exp_ret]
                            # Length mismatch guard
                            if len(coef) == len(x_vec):
                                z_adj = float(intercept) + sum(c * v for c, v in zip(coef, x_vec))
                                p_adj = float(_sigmoid(z_adj))
                                stacking['prob_forecast_adjusted'] = p_adj
                                stacking['prob_forecast_adjusted_components'] = {
                                    'intercept': intercept,
                                    'coef': coef,
                                    'x': x_vec,
                                    'z': z_adj,
                                }
                                try:
                                    log.debug(
                                        "[forecast_adj] symbol=%s base=%.4f x=%s z=%.4f p_adj=%.6f", 
                                        getattr(row, 'symbol', 'unknown'), float(sp), [round(v,6) for v in x_vec], z_adj, p_adj
                                    )
                                except Exception:
                                    pass
                                # expose final convenience field (can differ from smoothing/decision prob)
                                stacking['prob_final_adjusted'] = p_adj
                                # Preserve original final prob then override decision base
                                stacking['prob_final_base'] = stacking.get('prob_final')
                                base_final_before = stacking.get('prob_final')
                                # Delta clamp logic
                                clamp = float(getattr(_s, 'ADJUSTED_DELTA_CLAMP', 0.0))
                                fallback_mode = getattr(_s, 'ADJUSTED_DELTA_FALLBACK_MODE', 'limit')
                                if isinstance(base_final_before, (int, float)) and clamp > 0:
                                    delta_raw = p_adj - float(base_final_before)
                                    if abs(delta_raw) > clamp:
                                        if fallback_mode == 'revert':
                                            # keep base decision prob, still expose adjusted separately
                                            stacking['prob_final'] = float(base_final_before)
                                            stacking['prob_adjusted_delta_clamped'] = True
                                            stacking['prob_adjusted_delta_clamp_value'] = clamp
                                            stacking['prob_adjusted_delta_raw'] = delta_raw
                                        else:  # limit
                                            p_adj_limited = float(base_final_before) + (clamp if delta_raw > 0 else -clamp)
                                            stacking['prob_final'] = p_adj_limited
                                            stacking['prob_forecast_adjusted'] = p_adj_limited
                                            stacking['prob_final_adjusted'] = p_adj_limited
                                            stacking['prob_adjusted_delta_clamped'] = True
                                            stacking['prob_adjusted_delta_clamp_value'] = clamp
                                            stacking['prob_adjusted_delta_raw'] = delta_raw
                                            p_adj = p_adj_limited
                                        # record delta after clamp/revert decision base
                                    else:
                                        stacking['prob_final'] = p_adj
                                else:
                                    stacking['prob_final'] = p_adj
                                # Drift metrics between adjusted and base
                                try:
                                    base_final = stacking.get('prob_final_base')
                                    if isinstance(base_final, (int, float)) and base_final is not None:
                                        delta = p_adj - float(base_final)
                                        stacking['prob_adjusted_delta'] = delta
                                        self._adjust_delta_history.append(delta)
                                        # window means
                                        hist_list = list(self._adjust_delta_history)
                                        if len(hist_list) >= 5:
                                            import numpy as _np
                                            w50 = hist_list[-50:] if len(hist_list) >= 50 else hist_list
                                            w200 = hist_list[-200:] if len(hist_list) >= 200 else hist_list
                                            stacking['prob_adjusted_delta_mean_50'] = float(_np.mean(w50))
                                            stacking['prob_adjusted_delta_mean_200'] = float(_np.mean(w200))
                                            stacking['prob_adjusted_delta_abs_mean_200'] = float(_np.mean([abs(x) for x in w200]))
                                            # Divergence watchdog
                                            try:
                                                from .core.config import settings as _s2
                                                if len(hist_list) >= _s2.ADJUSTED_DIVERGENCE_MIN_SAMPLES:
                                                    mean200 = stacking.get('prob_adjusted_delta_mean_200')
                                                    abs_mean200 = stacking.get('prob_adjusted_delta_abs_mean_200')
                                                    if isinstance(mean200, (int,float)) and isinstance(abs_mean200, (int,float)):
                                                        divergence_flag = False
                                                        if abs(mean200) >= _s2.ADJUSTED_DIVERGENCE_MEAN_THRESHOLD or abs_mean200 >= _s2.ADJUSTED_DIVERGENCE_ABS_MEAN_THRESHOLD:
                                                            divergence_flag = True
                                                        if divergence_flag:
                                                            stacking['prob_adjusted_divergence_flag'] = True
                                                            stacking['prob_adjusted_divergence_mean_threshold'] = _s2.ADJUSTED_DIVERGENCE_MEAN_THRESHOLD
                                                            stacking['prob_adjusted_divergence_abs_mean_threshold'] = _s2.ADJUSTED_DIVERGENCE_ABS_MEAN_THRESHOLD
                                                            stacking['prob_adjusted_divergence_action'] = _s2.ADJUSTED_DIVERGENCE_ACTION
                                                            if _s2.ADJUSTED_DIVERGENCE_ACTION == 'revert' and 'prob_final_base' in stacking:
                                                                stacking['prob_final'] = float(base_final)
                                                                stacking['prob_final_adjusted_reverted'] = True
                                                                stacking['prob_adjusted_delta_reverted'] = delta
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                    else:
                        stacking['prob_forecast_adjusted'] = None
                    # Attach meta metrics if available
                    try:
                        if registry.stacking and getattr(registry.stacking, 'bvf_metrics', None):
                            stacking['bvf_meta_metrics'] = registry.stacking.bvf_metrics
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass

        # --- Precision-focused Entry Meta (optional) ---
        try:
            from .core.config import settings as _s3
            if stacking and stacking.get('ready') and getattr(registry, 'stacking', None) is not None and _s3.ENABLE_ENTRY_META:
                # Optional cooldown to limit recomputation
                import time as _t
                last_ts = getattr(self, '_entry_meta_last_ts', 0.0)
                min_iv = int(getattr(_s3, 'ENTRY_META_MIN_INTERVAL_SECONDS', 0))
                now_ts = _t.time()
                if min_iv <= 0 or (now_ts - float(last_ts) >= float(min_iv)):
                    coef = getattr(registry.stacking, 'entry_coef', None)
                    intercept = getattr(registry.stacking, 'entry_intercept', None)
                    if coef and intercept is not None:
                        # Build feature vector in order expected by training sidecar.
                        # Default convention (documented in _load_entry_meta):
                        #  [p_use, margin, room_to_forecast_min, rel_to_forecast_mean, forecast_expected_return]
                        p_base = stacking.get('prob_final')
                        p_adj = stacking.get('prob_final_adjusted')
                        p_use = None
                        try:
                            if getattr(_s3, 'ENTRY_META_USE_ADJUSTED_PROB', True) and isinstance(p_adj, (int, float)):
                                p_use = float(p_adj)
                            elif isinstance(p_base, (int, float)):
                                p_use = float(p_base)
                        except Exception:
                            p_use = None
                        margin = stacking.get('margin')
                        try:
                            margin = float(margin) if isinstance(margin, (int, float)) else 0.0
                        except Exception:
                            margin = 0.0
                        ff = stacking.get('forecast_features') or {}
                        room = ff.get('room_to_forecast_min') if isinstance(ff, dict) else None
                        relm = ff.get('rel_to_forecast_mean') if isinstance(ff, dict) else None
                        exr = ff.get('forecast_expected_return') if isinstance(ff, dict) else None
                        try:
                            room = float(room) if isinstance(room, (int, float)) else 0.0
                        except Exception:
                            room = 0.0
                        try:
                            relm = float(relm) if isinstance(relm, (int, float)) else 0.0
                        except Exception:
                            relm = 0.0
                        try:
                            exr = float(exr) if isinstance(exr, (int, float)) else 0.0
                        except Exception:
                            exr = 0.0
                        if isinstance(p_use, (int, float)):
                            x_vec = [float(p_use), margin, room, relm, exr]
                            # Guard for length mismatch: truncate/pad to match coef
                            if len(x_vec) != len(coef):
                                if len(x_vec) > len(coef):
                                    x_vec = x_vec[:len(coef)]
                                else:
                                    x_vec = x_vec + [0.0] * (len(coef) - len(x_vec))
                            z_meta = float(intercept) + sum(c * v for c, v in zip(coef, x_vec))
                            p_entry = float(_sigmoid(z_meta))
                            # Threshold precedence: dynamic > sidecar > env
                            # Prefer per-symbol dynamic threshold, else global dynamic
                            sym_key = str(getattr(row, 'symbol', self.symbol) or self.symbol).lower()
                            th_dyn_map = getattr(registry.stacking, 'entry_threshold_dynamic_by_symbol', None)
                            th_dyn = None
                            try:
                                if isinstance(th_dyn_map, dict):
                                    v = th_dyn_map.get(sym_key)
                                    if isinstance(v, (int, float)):
                                        th_dyn = float(v)
                            except Exception:
                                th_dyn = None
                            if th_dyn is None:
                                th_dyn = getattr(registry.stacking, 'entry_threshold_dynamic', None)
                            th_side = getattr(registry.stacking, 'entry_threshold_sidecar', None)
                            th_env = float(getattr(_s3, 'ENTRY_META_THRESHOLD', 0.90))
                            if isinstance(th_dyn, (int, float)):
                                th_eff = float(th_dyn)
                                th_src = 'dynamic'
                            elif isinstance(th_side, (int, float)):
                                th_eff = float(th_side)
                                th_src = 'sidecar'
                            else:
                                th_eff = th_env
                                th_src = 'env'
                            entry_decision = p_entry >= th_eff
                            stacking['entry_meta'] = {
                                'entry_prob': p_entry,
                                'entry_decision': entry_decision,
                                'threshold': th_eff,
                                'threshold_source': th_src,
                                'features_used': [round(v, 6) for v in x_vec],
                                'coef': coef,
                                'intercept': intercept,
                                'mode': 'precision',
                            }
                            self._entry_meta_last_ts = now_ts
                        else:
                            # Fallback to rule-of-thumb gating on final prob
                            p_final = stacking.get('prob_final')
                            th_eff = None
                            try:
                                th_eff = float(getattr(_s3, 'ENTRY_META_THRESHOLD', 0.90))
                            except Exception:
                                th_eff = 0.90
                            ok = isinstance(p_final, (int, float)) and float(p_final) >= max(float(stacking.get('threshold') or 0.0), th_eff)
                            stacking['entry_meta'] = {
                                'entry_prob': float(p_final) if isinstance(p_final, (int, float)) else None,
                                'entry_decision': bool(ok),
                                'threshold': th_eff,
                                'threshold_source': 'env',
                                'mode': 'fallback_rule',
                            }
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
