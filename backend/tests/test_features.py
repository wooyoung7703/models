import math

import pytest

from backend.app.features import FeatureCalculator
from backend.app.features_fast import FAST_FEATURE_KEYS

def test_feature_calculator_basic_sequence():
    fc = FeatureCalculator()
    # Construct an ascending price sequence to test run_up and RSI direction
    prices = [100 + i for i in range(15)]
    last_features = None
    for p in prices:
        last_features = fc.update(p, p+1, p-1, p, volume=1000)
    assert last_features is not None
    # run_up should reflect consecutive increases (at least > 0)
    assert last_features['run_up'] > 0
    # With only gains RSI logic may default to 50 due to zero losses; ensure no error and baseline >= 50
    assert last_features['rsi_14'] >= 50


def test_feature_calculator_volatility_and_macd():
    fc = FeatureCalculator()
    # Simulate oscillating prices to generate volatility and MACD values
    base = 100.0
    for i in range(50):
        close = base + (5 * (-1)**i)  # alternate swing
        fc.update(close, close+1, close-1, close, volume=500)
    feats = fc.update(base, base+1, base-1, base, volume=500)
    assert feats['vol_20'] >= 0
    assert 'macd' in feats and 'macd_signal' in feats and 'macd_hist' in feats


def test_fast_feature_path_matches_reference():
    fc_slow = FeatureCalculator(use_fast_features=False)
    fc_fast = FeatureCalculator(use_fast_features=True)

    for i in range(250):
        base = 100 + math.sin(i / 7.0) * 2 + i * 0.05
        open_ = base - 0.2
        close = base + 0.1
        high = close + 0.3
        low = open_ - 0.3
        volume = 500 + (i % 11) * 20
        feats_slow = fc_slow.update(open_, high, low, close, volume)
        feats_fast = fc_fast.update(open_, high, low, close, volume)

    # Ensure we have enough history for long windows before comparing
    assert len(fc_slow.closes) >= 200

    for key in FAST_FEATURE_KEYS:
        assert key in feats_slow, f"missing slow feature {key}"
        assert key in feats_fast, f"missing fast feature {key}"
        assert feats_fast[key] == pytest.approx(feats_slow[key], abs=1e-6)
