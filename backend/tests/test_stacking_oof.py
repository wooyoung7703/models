import os
from types import SimpleNamespace
from backend.app.training.stacking_oof import train as train_stacking_oof


def test_stacking_oof_minimal(tmp_path):
    out_path = tmp_path / 'stacking_oof_meta.json'
    args = SimpleNamespace(
        days=3,
        seq_len=16,
        interval='1m',
        batch_size=64,
        num_workers=0,
        device='cpu',
        amp=False,
        seed=42,
        past_window=15,
        future_window=60,
        min_gap=20,
        tolerance_pct=0.004,
        models=['lstm','tf','xgb'],
        folds=3,
        min_coverage=0.0,
        lstm_lr=1e-3,
        lstm_hidden_dim=32,
        tf_lr=1e-3,
        tf_model_dim=32,
        meta_out=str(out_path),
    )
    train_stacking_oof(args)
    assert os.path.exists(str(out_path))
    sidecar = str(out_path).replace('.json', '.metrics.json')
    assert os.path.exists(sidecar)
