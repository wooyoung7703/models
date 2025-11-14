import math
from collections import deque
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from .core.config import settings
import json
import os

class ProbabilityDriftMonitor:
    """Monitor probability distribution drift for stacking ensemble.

    Baseline distribution is loaded from stacking metrics sidecar (validation meta probabilities).
    Recent distribution is a rolling window of last N real-time stacking probabilities.

    Drift criteria:
      - KS test p-value < settings.PROB_DRIFT_KS_P_THRESHOLD OR
      - Wasserstein (mean absolute quantile diff) > settings.PROB_DRIFT_WASSERSTEIN_THRESHOLD

    Consecutive drift detections required: settings.PROB_DRIFT_CONSECUTIVE_REQUIRED
    """
    def __init__(self, meta_sidecar_path: Optional[str] = None):
        self.recent = deque(maxlen=settings.PROB_DRIFT_RECENT_SIZE)
        self.baseline_sample: List[float] = []
        self.baseline_loaded = False
        self.meta_sidecar_path = meta_sidecar_path or self._default_sidecar()
        self.last_state: Dict[str, Any] = {}
        self.consecutive_drift = 0
        self.load_baseline()

    def _default_sidecar(self) -> str:
        base, _ = os.path.splitext(settings.STACKING_META_PATH)
        return base + '.metrics.json'

    def load_baseline(self) -> None:
        path = self.meta_sidecar_path
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    payload = json.load(f)
                dist = payload.get('prob_dist') or {}
                sample = dist.get('sample') or []
                # Fallback: if explicit sample missing but quantiles present, synthesize from quantiles
                if not sample:
                    qmap = dist.get('quantiles') or {}
                    qs = sorted((float(k[1:])/100.0, v) for k, v in qmap.items() if k.startswith('q'))  # e.g., q05
                    sample = [float(v) for _, v in qs]
                self.baseline_sample = [float(x) for x in sample if isinstance(x, (int, float))]
                self.baseline_loaded = len(self.baseline_sample) >= settings.PROB_DRIFT_MIN_BASELINE
            else:
                self.baseline_loaded = False
        except Exception:
            self.baseline_loaded = False

    @staticmethod
    def _ks_2sample(x: List[float], y: List[float]) -> Tuple[float, float]:
        """Compute KS D statistic and approximate p-value (two-sided).
        Simple implementation without SciPy dependency.
        """
        if not x or not y:
            return 0.0, 1.0
        x_sorted = sorted(x); y_sorted = sorted(y)
        n = len(x_sorted); m = len(y_sorted)
        i = j = 0
        cdf_x = cdf_y = 0.0
        d = 0.0
        while i < n and j < m:
            if x_sorted[i] <= y_sorted[j]:
                cdf_x = (i + 1) / n
                i += 1
            else:
                cdf_y = (j + 1) / m
                j += 1
            d = max(d, abs(cdf_x - cdf_y))
        # remaining tails
        while i < n:
            cdf_x = (i + 1) / n
            i += 1
            d = max(d, abs(cdf_x - cdf_y))
        while j < m:
            cdf_y = (j + 1) / m
            j += 1
            d = max(d, abs(cdf_x - cdf_y))
        en = math.sqrt(n * m / (n + m))
        # Kolmogorov distribution approximation for p-value
        # p = 2 * sum_{k=1..inf} (-1)^{k-1} exp(-2 k^2 (d*en)^2)
        t = d * en
        p = 0.0
        for k in range(1, 6):  # truncate series
            p += ((-1)**(k-1)) * math.exp(-2 * (t**2) * (k**2))
        p *= 2.0
        p = max(0.0, min(1.0, p))
        return d, p

    @staticmethod
    def _wasserstein(x: List[float], y: List[float]) -> float:
        if not x or not y:
            return 0.0
        import numpy as np
        q = np.linspace(0, 1, 101)
        xq = np.quantile(x, q)
        yq = np.quantile(y, q)
        return float(np.mean(np.abs(xq - yq)))

    def update(self, prob: float) -> Dict[str, Any]:
        try:
            self.recent.append(float(prob))
        except Exception:
            return self.last_state
        recent_list = list(self.recent)
        ks_stat = None; ks_p = None; wass = None
        drift_flag = False
        if self.baseline_loaded and len(recent_list) >= max(10, settings.PROB_DRIFT_MIN_BASELINE // 2):
            ks_stat, ks_p = self._ks_2sample(recent_list, self.baseline_sample)
            wass = self._wasserstein(recent_list, self.baseline_sample)
            if (ks_p is not None and ks_p < settings.PROB_DRIFT_KS_P_THRESHOLD) or (wass is not None and wass > settings.PROB_DRIFT_WASSERSTEIN_THRESHOLD):
                drift_flag = True
        if drift_flag:
            self.consecutive_drift += 1
        else:
            self.consecutive_drift = 0
        state = {
            'timestamp_utc': datetime.now(tz=timezone.utc).isoformat(),
            'enabled': settings.PROB_DRIFT_ENABLED,
            'baseline_loaded': self.baseline_loaded,
            'baseline_size': len(self.baseline_sample),
            'recent_size': len(recent_list),
            'ks_stat': ks_stat,
            'ks_p_value': ks_p,
            'wasserstein': wass,
            'drift_flag': drift_flag,
            'consecutive_drift': self.consecutive_drift,
            'consecutive_required': settings.PROB_DRIFT_CONSECUTIVE_REQUIRED,
            'p_threshold': settings.PROB_DRIFT_KS_P_THRESHOLD,
            'wass_threshold': settings.PROB_DRIFT_WASSERSTEIN_THRESHOLD,
        }
        self.last_state = state
        return state


_drift_monitor: Optional[ProbabilityDriftMonitor] = None

def get_drift_monitor() -> Optional[ProbabilityDriftMonitor]:
    return _drift_monitor

def init_drift_monitor() -> Optional[ProbabilityDriftMonitor]:
    global _drift_monitor
    if not settings.PROB_DRIFT_ENABLED:
        return None
    if _drift_monitor is None:
        _drift_monitor = ProbabilityDriftMonitor()
    return _drift_monitor
