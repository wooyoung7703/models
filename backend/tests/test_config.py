from typing import Generator, List

import pytest

from backend.app.core import config


def _load_settings(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> config.Settings:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config.get_settings.cache_clear()
    return config.get_settings()


def _clear_env(monkeypatch: pytest.MonkeyPatch, keys: List[str]) -> None:
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_settings_env_override_and_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch, ["SYMBOL", "LOG_LEVEL", "LOG_FORMAT"])
    settings = _load_settings(
        monkeypatch,
        {
            "SYMBOL": "BTCUSDT",
            "LOG_LEVEL": "debug",
            "LOG_FORMAT": "PLAIN",
        },
    )
    assert settings.SYMBOL == "btcusdt"
    assert settings.SYMBOLS == ["btcusdt"]
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.LOG_FORMAT == "plain"


def test_settings_symbols_multi_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch, ["SYMBOL", "SYMBOLS"])
    settings = _load_settings(
        monkeypatch,
        {
            "SYMBOL": "xrpusdt",
            "SYMBOLS": "BTCUSDT, ETHUSDT , adaUSDT",
        },
    )
    assert settings.SYMBOLS == ["btcusdt", "ethusdt", "adausdt"]


def test_settings_retrain_defaults_follow_bottom_days(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(
        monkeypatch,
        [
            "BOTTOM_TRAIN_DAYS",
            "STACKING_META_DAYS",
            "RETRAIN_DAYS",
        ],
    )
    settings = _load_settings(
        monkeypatch,
        {
            "BOTTOM_TRAIN_DAYS": "60",
        },
    )
    assert settings.STACKING_META_DAYS == 60
    assert settings.RETRAIN_DAYS == 30


def test_settings_retrain_stacking_days_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch, ["RETRAIN_STACKING_DAYS", "STACKING_META_DAYS"])
    settings = _load_settings(
        monkeypatch,
        {
            "STACKING_META_DAYS": "50",
            "RETRAIN_STACKING_DAYS": "30",
        },
    )
    assert settings.RETRAIN_STACKING_DAYS == 30
    assert settings.STACKING_META_DAYS == 50


@pytest.fixture(autouse=True)
def _reset_cache() -> Generator[None, None, None]:
    """Ensure cached settings do not leak across tests."""
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()
