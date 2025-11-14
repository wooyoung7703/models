"""
Model adapter interfaces and lightweight loaders for real-time inference.

Notes:
- XGBoost adapter loads pickled booster and expected feature_names. If runtime cannot
  build a matching feature vector from Candle row, it will fill zeros and log a warning.
- LSTM/Transformer adapters are placeholders until sequence buffers and exact feature
  mapping are wired. They currently report not-ready.
- All adapters MUST fail soft (return None) to avoid breaking the real-time loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import logging
import os

from .core.config import settings

log = logging.getLogger(__name__)


@dataclass
class PredictResult:
    prob: Optional[float]
    ready: bool
    details: Dict[str, Any]


class BaseAdapter:
    name: str = "base"

    def ready(self) -> bool:
        return False

    def predict(self, row: Any, *, live_price: Optional[float] = None) -> PredictResult:
        return PredictResult(prob=None, ready=False, details={"reason": "not_implemented"})


class XGBTabularAdapter(BaseAdapter):
    name = "xgb"

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path or settings.MODEL_XGB_PATH
        self._booster = None
        self._features: List[str] = []
        self._xgb = None
        self._load()

    def _load(self) -> None:
        try:
            import pickle  # lazy
            try:
                import xgboost as xgb  # type: ignore
            except Exception as e:
                log.warning("[xgb] import failed: %s", e)
                return
            self._xgb = xgb
            if not os.path.exists(self.model_path):
                log.warning("[xgb] model file not found: %s (will stay not-ready)", self.model_path)
                return
            # Support two formats:
            # 1. Pickle dict { 'booster': Booster, 'features': [...] }
            # 2. Raw XGBoost json or binary model file (load directly)
            if self.model_path.endswith(('.pkl', '.pickle')):
                with open(self.model_path, "rb") as f:
                    payload = pickle.load(f)
                self._booster = payload.get("booster")
                self._features = list(payload.get("features") or [])
                if not self._features:
                    log.warning("[xgb] feature list missing in model payload; inference may be incorrect")
                log.info("[xgb] loaded pickled booster with %d features from %s", len(self._features), self.model_path)
            else:
                booster = xgb.Booster()
                booster.load_model(self.model_path)
                self._booster = booster
                self._features = []  # unknown; will fall back to getattr mapping or zeros
                log.info("[xgb] loaded raw model from %s", self.model_path)
        except Exception as e:
            log.warning("[xgb] load failed: %s", e)

    def ready(self) -> bool:
        return self._booster is not None and self._xgb is not None

    def _build_vector(self, row: Any) -> List[float]:
        # Attempt to build vector by attribute name; fallback zeros.
        vec: List[float] = []
        if not self._features:
            return [0.0] * 1
        for fname in self._features:
            # If training used generic names like f0,f1,... fill zeros
            if fname.startswith("f") and fname[1:].isdigit():
                vec.append(0.0)
                continue
            try:
                v = getattr(row, fname)
                if v is None:
                    v = 0.0
                vec.append(float(v))
            except Exception:
                vec.append(0.0)
        return vec

    def predict(self, row: Any, *, live_price: Optional[float] = None) -> PredictResult:
        if not self.ready():
            return PredictResult(prob=None, ready=False, details={"reason": "not_ready"})
        try:
            import numpy as np  # lazy
            x = self._build_vector(row)
            # self._xgb may be a module; guard for safety
            if self._xgb is None:
                raise RuntimeError("xgboost module not loaded")
            dmat = self._xgb.DMatrix(np.asarray([x], dtype=float), feature_names=self._features if self._features else None)  # type: ignore[attr-defined]
            if self._booster is None:
                raise RuntimeError("booster not loaded")
            p = float(self._booster.predict(dmat)[0])  # type: ignore[attr-defined]
            return PredictResult(prob=p, ready=True, details={"features": len(self._features)})
        except Exception as e:
            log.warning("[xgb] predict failed: %s", e)
            return PredictResult(prob=None, ready=False, details={"error": str(e)})


class LSTMSeqAdapter(BaseAdapter):
    name = "lstm"

    def __init__(self, model_path: Optional[str] = None, seq_len: Optional[int] = None) -> None:
        self.model_path = model_path or settings.MODEL_LSTM_PATH
        self.seq_len = int(seq_len or settings.SEQ_LEN)
        self._torch = None
        self._model = None
        self._feature_dim: Optional[int] = None
        self._load()

    def ready(self) -> bool:
        return self._model is not None

    def predict(self, row: Any, *, live_price: Optional[float] = None) -> PredictResult:
        if not self.ready():
            return PredictResult(prob=None, ready=False, details={"reason": "model_not_loaded"})
        from .seq_buffer import get_buffer
        try:
            buf = get_buffer(getattr(row, 'symbol', 'unknown'))
            if len(buf) < settings.SEQ_MIN_READY:
                return PredictResult(prob=None, ready=False, details={"reason": "insufficient_sequence", "have": len(buf)})
            import torch
            seq = buf.to_list()[-self.seq_len:]
            x = torch.tensor([seq], dtype=torch.float32)
            # Pad sequence vectors to expected feature_dim if checkpoint expects more features
            if self._feature_dim is not None and x.size(-1) < self._feature_dim:
                pad = torch.zeros((x.size(0), x.size(1), self._feature_dim - x.size(-1)))
                x = torch.cat([x, pad], dim=-1)
            model = self._model
            if model is None:
                return PredictResult(prob=None, ready=False, details={"reason": "model_none"})
            model.eval()
            with torch.no_grad():
                out = model(x)
            prob = float(out.squeeze().item())
            # Clamp probability if model output is logit
            if prob < 0.0 or prob > 1.0:
                import math
                prob = 1/(1+math.exp(-prob))
            return PredictResult(prob=prob, ready=True, details={"seq_len": len(seq), "feature_dim": self._feature_dim, "padded": self._feature_dim is not None and self._feature_dim > len(seq[0])})
        except Exception as e:
            return PredictResult(prob=None, ready=False, details={"error": str(e)})

    def _load(self) -> None:
        """Load LSTM model from checkpoint dict with state_dict.

        Training artifacts save format:
        torch.save({'state_dict': model.state_dict(), 'config': vars(args), 'epoch': epoch}, path)
        """
        import os
        if not os.path.exists(self.model_path):
            log.warning("[lstm] model path not found: %s", self.model_path)
            return
        try:
            import torch  # type: ignore
        except Exception as e:
            log.warning("[lstm] torch import failed, skipping load: %s", e)
            return
        self._torch = torch
        ckpt = None
        try:
            ckpt = torch.load(self.model_path, map_location='cpu')
        except Exception as e:
            log.debug("[lstm] torch.load failed (%s), will try jit.load", e)
        if isinstance(ckpt, dict) and ('state_dict' in ckpt):
            cfg = ckpt.get('config') or {}
            hidden_dim = int(cfg.get('hidden_dim', 64))
            num_layers = int(cfg.get('num_layers', 1))
            dropout = float(cfg.get('dropout', 0.0))
            mode = str(cfg.get('mode', 'cls_bottom'))
            # Infer feature_dim from checkpoint weight if available (lstm.weight_ih_l0 shape: (4*hidden, feature_dim))
            feature_dim = 6
            try:
                w = ckpt['state_dict'].get('lstm.weight_ih_l0')
                if w is not None and hasattr(w, 'shape') and len(w.shape) == 2:
                    feature_dim = int(w.shape[1])
            except Exception:
                pass
            feature_list = cfg.get('feature_list') or []
            self._feature_list = list(feature_list) if isinstance(feature_list, (list, tuple)) else []
            class _LSTMModel(torch.nn.Module):
                def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int, dropout: float, mode: str):
                    super().__init__()
                    self.mode = mode
                    self.lstm = torch.nn.LSTM(feature_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
                    self.head = torch.nn.Linear(hidden_dim, 1)
                def forward(self, x):
                    out, _ = self.lstm(x)
                    last = out[:, -1, :]
                    out = self.head(last).squeeze(-1)
                    return out
            model = _LSTMModel(feature_dim=feature_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, mode=mode)
            try:
                missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
                if missing:
                    log.debug("[lstm] missing keys: %s", missing)
                if unexpected:
                    log.debug("[lstm] unexpected keys: %s", unexpected)
                self._model = model.cpu()
                self._feature_dim = feature_dim
                log.info("[lstm] loaded state_dict model (hidden=%d,layers=%d,dropout=%.2f,mode=%s,feature_dim=%d) from %s", hidden_dim, num_layers, dropout, mode, feature_dim, self.model_path)
                if self._feature_list and len(self._feature_list) != feature_dim:
                    log.warning("[lstm] feature_list length (%d) != inferred feature_dim (%d)", len(self._feature_list), feature_dim)
            except Exception as e:
                log.warning("[lstm] state_dict load failed: %s", e)
        else:
            try:
                self._model = torch.jit.load(self.model_path)
                log.info("[lstm] loaded TorchScript model from %s", self.model_path)
            except Exception as e:
                log.warning("[lstm] load failed: %s", e)


class TFSeqAdapter(BaseAdapter):
    name = "tf"

    def __init__(self, model_path: Optional[str] = None, seq_len: Optional[int] = None) -> None:
        self.model_path = model_path or settings.MODEL_TRANSFORMER_PATH
        self.seq_len = int(seq_len or settings.SEQ_LEN)
        self._torch = None
        self._model = None
        self._feature_dim: Optional[int] = None
        self._load()

    def ready(self) -> bool:
        return self._model is not None

    def predict(self, row: Any, *, live_price: Optional[float] = None) -> PredictResult:
        if not self.ready():
            return PredictResult(prob=None, ready=False, details={"reason": "model_not_loaded"})
        from .seq_buffer import get_buffer
        try:
            buf = get_buffer(getattr(row, 'symbol', 'unknown'))
            if len(buf) < settings.SEQ_MIN_READY:
                return PredictResult(prob=None, ready=False, details={"reason": "insufficient_sequence", "have": len(buf)})
            import torch
            seq = buf.to_list()[-self.seq_len:]
            x = torch.tensor([seq], dtype=torch.float32)
            if self._feature_dim is not None and x.size(-1) < self._feature_dim:
                pad = torch.zeros((x.size(0), x.size(1), self._feature_dim - x.size(-1)))
                x = torch.cat([x, pad], dim=-1)
            model = self._model
            if model is None:
                return PredictResult(prob=None, ready=False, details={"reason": "model_none"})
            model.eval()
            with torch.no_grad():
                out = model(x)
            prob = float(out.squeeze().item())
            if prob < 0.0 or prob > 1.0:
                import math
                prob = 1/(1+math.exp(-prob))
            return PredictResult(prob=prob, ready=True, details={"seq_len": len(seq), "feature_dim": self._feature_dim, "padded": self._feature_dim is not None and self._feature_dim > len(seq[0])})
        except Exception as e:
            return PredictResult(prob=None, ready=False, details={"error": str(e)})

    def _load(self) -> None:
        """Load Transformer model from state_dict checkpoint; fallback to TorchScript if needed."""
        import os
        if not os.path.exists(self.model_path):
            log.warning("[tf] model path not found: %s", self.model_path)
            return
        try:
            import torch  # type: ignore
        except Exception as e:
            log.warning("[tf] torch import failed, skipping load: %s", e)
            return
        self._torch = torch
        ckpt = None
        try:
            ckpt = torch.load(self.model_path, map_location='cpu')
        except Exception as e:
            log.debug("[tf] torch.load failed (%s), will try jit.load", e)
        if isinstance(ckpt, dict) and ('state_dict' in ckpt):
            cfg = ckpt.get('config') or {}
            model_dim = int(cfg.get('model_dim', 64))
            nhead = int(cfg.get('nhead', 4))
            num_layers = int(cfg.get('num_layers', 1))
            dropout = float(cfg.get('dropout', 0.1))
            mode = str(cfg.get('mode', 'cls_bottom'))
            feature_dim = 6
            try:
                w = ckpt['state_dict'].get('in_proj.weight')
                if w is not None and hasattr(w, 'shape') and len(w.shape) == 2:
                    feature_dim = int(w.shape[1])
            except Exception:
                pass
            feature_list = cfg.get('feature_list') or []
            self._feature_list = list(feature_list) if isinstance(feature_list, (list, tuple)) else []
            class _PositionalEncoding(torch.nn.Module):
                def __init__(self, d_model: int, max_len: int = 1000):
                    super().__init__()
                    pe = torch.zeros(max_len, d_model)
                    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-(torch.log(torch.tensor(10000.0)) / d_model)))
                    pe[:, 0::2] = torch.sin(position * div_term)
                    pe[:, 1::2] = torch.cos(position * div_term)
                    pe = pe.unsqueeze(0)  # (1, max_len, d_model)
                    self.register_buffer('pe', pe)
                def forward(self, x):
                    # Use registered buffer directly and clone slice to avoid static analyzer complaints
                    import torch as _torch
                    pe_full = getattr(self, 'pe')  # (1, max_len, d_model)
                    t = x.size(1)
                    pe_slice = pe_full.index_select(1, _torch.arange(t))  # (1, T, d_model)
                    return x + pe_slice
            class _TransformerModel(torch.nn.Module):
                def __init__(self, feature_dim: int, model_dim: int, nhead: int, num_layers: int, dropout: float, mode: str):
                    super().__init__()
                    self.mode = mode
                    self.in_proj = torch.nn.Linear(feature_dim, model_dim)
                    encoder_layer = torch.nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=model_dim*4, dropout=dropout, batch_first=True)
                    self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    self.pos_enc = _PositionalEncoding(model_dim)
                    self.norm = torch.nn.LayerNorm(model_dim)
                    self.head = torch.nn.Linear(model_dim, 1)
                def forward(self, x):
                    x = self.in_proj(x)
                    x = self.pos_enc(x)
                    x = self.encoder(x)
                    x = self.norm(x)
                    pooled = x[:, -1, :]
                    out = self.head(pooled).squeeze(-1)
                    return out
            model = _TransformerModel(feature_dim=feature_dim, model_dim=model_dim, nhead=nhead, num_layers=num_layers, dropout=dropout, mode=mode)
            try:
                missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
                if missing:
                    log.debug("[tf] missing keys: %s", missing)
                if unexpected:
                    log.debug("[tf] unexpected keys: %s", unexpected)
                self._model = model.cpu()
                self._feature_dim = feature_dim
                log.info("[tf] loaded state_dict model (d=%d,nhead=%d,layers=%d,dropout=%.2f,mode=%s,feature_dim=%d) from %s", model_dim, nhead, num_layers, dropout, mode, feature_dim, self.model_path)
                if self._feature_list and len(self._feature_list) != feature_dim:
                    log.warning("[tf] feature_list length (%d) != inferred feature_dim (%d)", len(self._feature_list), feature_dim)
            except Exception as e:
                log.warning("[tf] state_dict load failed: %s", e)
        else:
            try:
                self._model = torch.jit.load(self.model_path)
                log.info("[tf] loaded TorchScript model from %s", self.model_path)
            except Exception as e:
                log.warning("[tf] load failed: %s", e)


class StackingCombiner:
    """Combine base probabilities using stacking meta config.

    Supported methods:
      - logistic: sigmoid(intercept + sum_i coef_i * logit(p_i))
      - dynamic: weights over logit(p_i) -> sigmoid(sum)
    """
    def __init__(self, meta_path: Optional[str] = None) -> None:
        self.meta_path = meta_path or settings.STACKING_META_PATH
        self.method: str = "logistic"
        self.model_order: List[str] = []
        self.coef: Optional[List[float]] = None
        self.intercept: Optional[float] = None
        self.dynamic_weights: Optional[Dict[str, float]] = None
        self.best_threshold: Optional[float] = None
        self.double_t_low: Optional[float] = None
        self.double_t_high: Optional[float] = None
        # calibration params
        self.calibration = {}
        # regime-aware dynamic weights: {"low_vol": {model: weight, ...}, "high_vol": {...}}
        self.regime_weights = None
        # bayesian weights (if ensemble == 'bayes')
        self.bayes_weights: Optional[Dict[str, float]] = None
        self._loaded = False
        self._load()

    def _load(self) -> None:
        try:
            if not os.path.exists(self.meta_path):
                log.warning("[stacking] meta file not found: %s", self.meta_path)
                return
            with open(self.meta_path, "r") as f:
                meta = json.load(f)
            self.method = meta.get("ensemble", "logistic")
            self.model_order = meta.get("model_order") or meta.get("models") or []
            self.coef = meta.get("coef")
            self.intercept = meta.get("intercept")
            self.dynamic_weights = meta.get("dynamic_weights")
            self.bayes_weights = meta.get("bayes_weights")
            self.calibration = meta.get("calibration") or {}
            self.regime_weights = meta.get("regime_weights")
            # Try to load sidecar metrics for best threshold hints
            try:
                base, _ = os.path.splitext(self.meta_path)
                sidecar = base + ".metrics.json"
                if os.path.exists(sidecar):
                    with open(sidecar, "r") as sf:
                        m = json.load(sf)
                    bt = m.get("best_threshold_precision")
                    if isinstance(bt, (int, float)):
                        self.best_threshold = float(bt)
                    dl = m.get("double_t_low"); dh = m.get("double_t_high")
                    if isinstance(dl, (int, float)):
                        self.double_t_low = float(dl)
                    if isinstance(dh, (int, float)):
                        self.double_t_high = float(dh)
            except Exception:
                pass
            self._loaded = True
            log.info("[stacking] loaded meta (method=%s models=%s)", self.method, ",".join(self.model_order))
        except Exception as e:
            log.warning("[stacking] load failed: %s", e)

    def ready(self) -> bool:
        return self._loaded and bool(self.model_order)

    @staticmethod
    def _sigmoid(x: float) -> float:
        import math
        try:
            if x >= 0:
                z = math.exp(-x)
                return 1.0 / (1.0 + z)
            else:
                z = math.exp(x)
                return z / (1.0 + z)
        except Exception:
            return 0.5

    @staticmethod
    def _logit(p: float) -> float:
        import math
        eps = 1e-9
        p = min(max(p, eps), 1 - eps)
        return math.log(p / (1 - p))

    def combine(self, probs: Dict[str, Optional[float]], *, threshold: Optional[float] = None, regime: Optional[str] = None) -> Dict[str, Any]:
        if not self.ready():
            return {"ready": False, "reason": "meta_not_ready"}
        vals: List[float] = []
        used: List[str] = []
        for name in self.model_order:
            p = probs.get(name)
            if p is None:
                continue
            try:
                vals.append(self._logit(float(p)))
                used.append(name)
            except Exception:
                continue
        if not vals:
            return {"ready": False, "reason": "no_base_probs"}
        z = 0.0  # raw logit ensemble value before sigmoid
        method = self.method  # working method (may switch to dynamic_regime or override)
        # Optional override for rollback/risk control
        try:
            override = getattr(settings, 'STACKING_OVERRIDE_METHOD', None)
            if isinstance(override, str) and override.strip():
                method = override.strip().lower()
        except Exception:
            pass
        # Regime weighting has highest precedence if available & requested
        if regime in ("low_vol", "high_vol") and self.regime_weights:
            try:
                rW = self.regime_weights.get(regime) or {}
                weights = [float(rW.get(m, 0.0)) for m in used]
                s = sum(weights)
                if s > 0:
                    z = sum((w / s) * v for w, v in zip(weights, vals))
                    method = "dynamic_regime"
            except Exception:
                pass
        if method == "logistic" and self.coef is not None and self.intercept is not None and z == 0.0:
            # align coef to used order subset if needed
            try:
                # Map model_order -> coef; then pick only used
                weights = {m: float(w) for m, w in zip(self.model_order, self.coef)}
                z = float(self.intercept) + sum(weights[m] * v for m, v in zip(used, vals))
            except Exception:
                method = "mean"
        elif method == "dynamic" and self.dynamic_weights is not None and z == 0.0:
            try:
                weights = [float(self.dynamic_weights.get(m, 0.0)) for m in used]
                s = sum(weights)
                if s <= 0:
                    method = "mean"
                else:
                    z = sum((w / s) * v for w, v in zip(weights, vals))
            except Exception:
                method = "mean"
        elif method == "bayes" and self.bayes_weights is not None and z == 0.0:
            try:
                weights = [float(self.bayes_weights.get(m, 0.0)) for m in used]
                s = sum(weights)
                if s <= 0:
                    method = "mean"
                else:
                    z = sum((w / s) * v for w, v in zip(weights, vals))
            except Exception:
                method = "mean"
        if method == "mean":
            z = sum(vals) / max(1, len(vals))
        prob = self._sigmoid(z)
        raw_prob = prob
        # Apply optional calibration (Platt scaling or isotonic piecewise) after ensemble prob
        if settings.ENABLE_CALIBRATION and self.calibration:
            try:
                method = self.calibration.get("method")
                if method == "platt":
                    a = float(self.calibration.get("a", 1.0))
                    b = float(self.calibration.get("b", 0.0))
                    # Platt scaling expects logit transform then sigmoid(a*logit + b)
                    logit_p = self._logit(max(1e-9, min(1-1e-9, raw_prob)))
                    prob = self._sigmoid(a * logit_p + b)
                elif method == "isotonic":
                    # calibration points: list of [x,p] sorted by x in [0,1]
                    pts = self.calibration.get("points") or []
                    if isinstance(pts, list) and len(pts) >= 2:
                        x = raw_prob
                        # linear interpolate between nearest points
                        prev_x, prev_y = pts[0]
                        for cur_x, cur_y in pts[1:]:
                            if x <= cur_x:
                                # interpolate
                                if cur_x == prev_x:
                                    prob = float(cur_y)
                                else:
                                    t = (x - prev_x) / (cur_x - prev_x)
                                    prob = float(prev_y + t * (cur_y - prev_y))
                                break
                            prev_x, prev_y = cur_x, cur_y
                        else:
                            prob = float(pts[-1][1])
            except Exception:
                prob = raw_prob
        # threshold preference: explicit arg > settings override >=0 > sidecar best_threshold
        th = threshold
        th_source: Optional[str] = None
        if th is not None:
            th_source = "explicit"
        else:
            if settings.STACKING_THRESHOLD >= 0:
                th = settings.STACKING_THRESHOLD
                th_source = "env"
            elif self.best_threshold is not None:
                th = self.best_threshold
                th_source = "sidecar"
        decision = (prob >= th) if th is not None else None
        return {
            "ready": True,
            "method": method,
            "used_models": used,
            "prob": prob,
            "raw_prob": raw_prob,
            "threshold": th,
            "threshold_source": th_source,
            "decision": decision,
            "z": z,
            "calibrated": settings.ENABLE_CALIBRATION and bool(self.calibration),
            "cal_method": self.calibration.get("method") if self.calibration else None,
            "regime_used": regime if (self.regime_weights is not None and regime in ("low_vol","high_vol")) else None,
        }


# Global registry helpers
class ModelRegistry:
    def __init__(self) -> None:
        self.adapters: Dict[str, BaseAdapter] = {}
        self.stacking: Optional[StackingCombiner] = None

    def load_from_settings(self) -> None:
        if settings.ENABLE_BASE_MODELS:
            # XGB is the most reliable to start with
            try:
                self.adapters["xgb"] = XGBTabularAdapter()
            except Exception as e:
                log.warning("[registry] xgb init failed: %s", e)
            # Placeholders for seq models (will be wired later)
            self.adapters.setdefault("lstm", LSTMSeqAdapter())
            self.adapters.setdefault("tf", TFSeqAdapter())
        if settings.ENABLE_STACKING:
            try:
                self.stacking = StackingCombiner()
            except Exception as e:
                log.warning("[registry] stacking init failed: %s", e)

    def status(self) -> Dict[str, Any]:
        base_ready = {name: adapter.ready() for name, adapter in self.adapters.items()}
        # Build lightweight details for transparency
        details: Dict[str, Any] = {}
        for name, adapter in self.adapters.items():
            info: Dict[str, Any] = {"ready": adapter.ready()}
            try:
                # try to expose feature_dim if present (seq models)
                fdim = getattr(adapter, "_feature_dim", None)
                if fdim is not None:
                    info["feature_dim"] = int(fdim)
                # expose seq_len expectation if present
                slen = getattr(adapter, "seq_len", None)
                if slen is not None:
                    info["seq_len"] = int(slen)
                # xgb features if available
                if name == "xgb":
                    feats = getattr(adapter, "_features", None)
                    if feats is not None:
                        info["n_features"] = len(feats)
                flist = getattr(adapter, "_feature_list", None)
                if isinstance(flist, list) and flist:
                    info["feature_list_len"] = len(flist)
            except Exception:
                pass
            details[name] = info
        stk: Dict[str, Any] = {
            "ready": (self.stacking.ready() if self.stacking else False),
        }
        try:
            if self.stacking and self.stacking.ready():
                # Compute threshold selection logic (same precedence as combine) for observability
                th: Optional[float] = None
                th_source: Optional[str] = None
                if settings.STACKING_THRESHOLD >= 0:
                    th = settings.STACKING_THRESHOLD
                    th_source = "env"
                elif self.stacking.best_threshold is not None:
                    th = self.stacking.best_threshold
                    th_source = "sidecar"
                # effective method respects override if set
                method_meta = self.stacking.method
                override = getattr(settings, 'STACKING_OVERRIDE_METHOD', '')
                method_effective = override if override else method_meta
                stk.update({
                    "method": method_effective,
                    "method_meta": method_meta,
                    "method_override": override or None,
                    "models": self.stacking.model_order,
                    "threshold": th,
                    "threshold_source": th_source,
                })
        except Exception:
            pass
        return {
            "base_models": base_ready,
            "stacking": stk,
            "details": details,
        }


# Singleton instance to be used by app
registry = ModelRegistry()
