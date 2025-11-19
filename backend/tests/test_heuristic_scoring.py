from types import SimpleNamespace

from backend.app.heuristic_scoring import compute_bottom_score


def make_row(**kwargs):
    base = {
        "close": 100.0,
        "rsi_14": 50.0,
        "bb_pct_b_20_2": 0.5,
        "drawdown_from_max_20": 0.0,
        "macd_hist": 0.0,
        "vol_z_20": 0.0,
        "williams_r_14": -50.0,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_compute_bottom_score_range():
    row = make_row()
    result = compute_bottom_score(row)
    assert 0.0 <= result.prob <= 1.0
    assert "s_rsi" in result.components
    assert "logit" in result.components


def test_stronger_bottom_signal_increases_prob():
    neutral = compute_bottom_score(make_row())
    oversold = compute_bottom_score(
        make_row(
            rsi_14=20.0,
            bb_pct_b_20_2=0.05,
            drawdown_from_max_20=-0.04,
            vol_z_20=2.5,
            williams_r_14=-95.0,
        )
    )
    assert oversold.prob > neutral.prob


def test_price_override_keeps_components():
    row = make_row(close=90.0)
    result = compute_bottom_score(row, price_override=80.0)
    assert result.components["close"] == 80.0
