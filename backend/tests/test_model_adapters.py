import json
import pytest

from backend.app import model_adapters


def test_analyze_feature_alignment_success():
    features = [
        "close",
        "rsi_14",
        "bb_pct_b_20_2",
    ]
    stats = model_adapters.analyze_feature_alignment(features)
    assert stats["total"] == len(features)
    assert stats["matched"] == len(features)
    assert stats["generic"] == 0


def test_analyze_feature_alignment_unknown():
    with pytest.raises(ValueError):
        model_adapters.analyze_feature_alignment(["not_a_feature"])


def test_load_feature_manifest(tmp_path):
    manifest = tmp_path / "features.json"
    payload = {"feature_names": ["close", "rsi_14"]}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    names = model_adapters._load_feature_manifest(str(manifest))
    assert names == payload["feature_names"]
