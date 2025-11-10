from backend.app.features import FeatureCalculator

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
