import json
from pathlib import Path

import pytest

from backend.app.model_adapters import StackingCombiner
from backend.app.core.config import settings


def test_stacking_loads_valid_meta(tmp_path):
    meta = {
        "version": 1,
        "ensemble": "logistic",
        "model_order": ["lstm", "xgb"],
        "coef": [0.4, 0.6],
        "intercept": -1.0,
    }
    target = tmp_path / "stacking_meta.json"
    target.write_text(json.dumps(meta))
    comb = StackingCombiner(meta_path=str(target))
    assert comb.ready()
    assert comb.model_order == ["lstm", "xgb"]


def test_stacking_schema_rejects_invalid_meta(tmp_path):
    meta = {
        "version": 1,
        "ensemble": "logistic",
        "model_order": [],  # invalid due to minItems
        "coef": [0.4, 0.6],
        "intercept": -1.0,
    }
    target = tmp_path / "stacking_meta.json"
    target.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        StackingCombiner(meta_path=str(target))
