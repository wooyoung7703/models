import os
from types import SimpleNamespace

from backend.app.training.train_transformer import train as train_transformer

def test_train_transformer_minimal(tmp_path):
    out_path = tmp_path / 'transformer_model_test.pt'
    args = SimpleNamespace(
        days=7,
        seq_len=16,
        val_ratio=0.2,
        batch_size=64,
        num_workers=0,
        model_dim=32,
        nhead=4,
        num_layers=1,
        dropout=0.1,
        lr=1e-3,
        epochs=1,
        mode='cls_bottom',
        interval='1m',
        model_out=str(out_path),
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
    train_transformer(args)
    assert os.path.exists(str(out_path))
    sidecar = str(out_path).replace('.pt', '.metrics.json')
    assert os.path.exists(sidecar)
