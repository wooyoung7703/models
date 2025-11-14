import os
import json
import types

# Set env BEFORE importing app to influence settings/registry
os.environ['DISABLE_BACKGROUND_LOOPS'] = '1'
os.environ['GAP_FILL_ENABLED'] = '0'
os.environ['ENABLE_STACKING'] = '1'
os.environ['ENABLE_BASE_MODELS'] = '1'

from fastapi.testclient import TestClient
from backend.app.main import app  # noqa: E402
from backend.app.model_adapters import StackingCombiner, registry as _reg  # noqa: E402
from backend.app.core.config import settings  # noqa: E402

def test_stacking_threshold_precedence(tmp_path):
    # Create a fake stacking meta + sidecar metrics
    meta_path = tmp_path / 'stacking_meta.json'
    meta = {
        "ensemble": "logistic",
        "model_order": ["lstm", "tf", "xgb"],
        "coef": [0.5, 0.25, 0.25],
        "intercept": 0.0,
    }
    meta_path.write_text(json.dumps(meta))
    sidecar_metrics = {"best_threshold_precision": 0.7, "double_t_low": 0.3, "double_t_high": 0.9}
    (tmp_path / 'stacking_meta.metrics.json').write_text(json.dumps(sidecar_metrics))

    # Case 1: explicit argument overrides everything
    comb = StackingCombiner(meta_path=str(meta_path))
    probs = {"lstm": 0.8, "tf": 0.6, "xgb": 0.4}
    out_explicit = comb.combine(probs, threshold=0.85)
    assert out_explicit['threshold'] == 0.85
    assert out_explicit['threshold_source'] == 'explicit'

    # Case 2: env override when >=0 and no explicit
    old_env = settings.STACKING_THRESHOLD
    try:
        settings.STACKING_THRESHOLD = 0.9
        out_env = comb.combine(probs)
        assert out_env['threshold'] == 0.9
        assert out_env['threshold_source'] == 'env'
    finally:
        settings.STACKING_THRESHOLD = old_env

    # Case 3: sidecar best_threshold when no explicit/env override
    settings.STACKING_THRESHOLD = -1
    out_sidecar = comb.combine(probs)
    assert out_sidecar['threshold'] == 0.7
    assert out_sidecar['threshold_source'] == 'sidecar'


def test_nowcast_meta_injection():
    # Ensure registry thinks stacking is ready during this test
    if _reg.stacking is None or not _reg.stacking.ready():
        _reg.stacking = StackingCombiner(meta_path=settings.STACKING_META_PATH)
    # Inject artificial nowcast state mimicking predictor output
    app.state.latest_nowcast = {
        'xrpusdt': {
            'symbol': 'xrpusdt',
            'interval': '1m',
            'timestamp': '2025-01-01T00:00:00',
            'price': 1.2345,
            'price_source': 'live',
            'bottom_score': 0.42,
            'components': {},
            'base_probs': { 'lstm': 0.33, 'tf': 0.44, 'xgb': 0.55 },
            'stacking': {
                'ready': True,
                'method': 'logistic',
                'used_models': ['lstm','tf','xgb'],
                'prob': 0.52,
                'threshold': 0.9,
                'threshold_source': 'env',
                'decision': False
            }
        }
    }
    client = TestClient(app)
    r = client.get('/nowcast')
    assert r.status_code == 200
    data = r.json()
    assert '_stacking_meta' in data, 'Expected _stacking_meta in aggregated nowcast result'
    meta = data['_stacking_meta']
    assert meta['method'] == 'logistic'
    assert meta['threshold'] == 0.9
    assert meta['used_models'] == ['lstm','tf','xgb']
