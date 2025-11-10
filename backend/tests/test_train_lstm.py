import os
from types import SimpleNamespace

from backend.app.training.train_lstm import train as train_lstm

def test_train_lstm_minimal(tmp_path):
    out_path = tmp_path / 'lstm_bottom_synth_test.pt'
    args = SimpleNamespace(
        days=7,
        seq_len=16,
        val_ratio=0.2,
        batch_size=64,
        num_workers=0,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
        lr=1e-3,
        epochs=1,
        model_out=str(out_path),
        mode='cls_bottom',
        interval='1m',
        device='cpu',
        amp=False,
        scheduler='none',
        grad_clip=0.0,
        patience=2,
        early_stop_metric='f1',
        seed=42,
        past_window=15,
        future_window=60,
        min_gap=20,
        tolerance_pct=0.004,
        pos_weight=-1.0,
    )
    train_lstm(args)
    assert os.path.exists(str(out_path))
    sidecar = str(out_path).replace('.pt', '.metrics.json')
    assert os.path.exists(sidecar)
