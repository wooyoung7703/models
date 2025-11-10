import os
from backend.app.training.stacking import train as train_stacking
from types import SimpleNamespace


def test_stacking_minimal(tmp_path):
    out_path = tmp_path / 'stacking_meta.json'
    args = SimpleNamespace(
        days=3,
        seq_len=16,
        val_ratio=0.2,
        batch_size=64,
        num_workers=0,
        interval='1m',
        device='cpu',
        amp=False,
        seed=42,
        past_window=15,
        future_window=60,
        min_gap=20,
        tolerance_pct=0.004,
        models=['lstm','tf','xgb'],
        lstm_epochs=1,
        lstm_hidden_dim=32,
        lstm_lr=1e-3,
        tf_epochs=1,
        tf_model_dim=32,
        tf_lr=1e-3,
        xgb_estimators=10,
        min_coverage=0.0,
        meta_out=str(out_path),
    )
    train_stacking(args)
    assert os.path.exists(str(out_path))
    sidecar = str(out_path).replace('.json', '.metrics.json')
    assert os.path.exists(sidecar)
