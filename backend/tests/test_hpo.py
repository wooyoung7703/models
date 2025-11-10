import os
from types import SimpleNamespace

from backend.app.training.hpo import run_study

def test_hpo_lstm_minimal(tmp_path):
    args = SimpleNamespace(
        model='lstm',
        mode='cls_bottom',
        interval='1m',
        days=3,
        seq_len=16,
        val_ratio=0.2,
        trials=1,
        timeout=0,
        epochs=1,
        metric='f1',
        output_dir=str(tmp_path),
        seed=42,
        device='cpu',
        past_window=15,
        future_window=60,
        min_gap=20,
        tolerance_pct=0.004,
    )
    res = run_study(args)
    assert 'best_params' in res
    assert os.path.exists(os.path.join(str(tmp_path), 'best_params.json'))
