import os
import csv
from types import SimpleNamespace

from backend.app.training.stacking_oof import train as train_stacking_oof
from backend.app.training.stacking_infer import infer


def _write_prob_csv(path, n, base):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id','prob'])
        for i in range(n):
            # simple deterministic pattern
            p = min(0.99, max(0.01, base + (i - n/2) * 0.001))
            w.writerow([i, f"{p:.4f}"])


def test_stacking_infer_with_oof_meta(tmp_path):
    # First, train a tiny OOF meta to get a real meta_config
    meta_path = tmp_path / 'stacking_oof_meta.json'
    args_oof = SimpleNamespace(
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
        meta_out=str(meta_path),
    )
    train_stacking_oof(args_oof)
    assert os.path.exists(str(meta_path))

    # Prepare base probability CSVs with matching model names
    lstm_csv = tmp_path / 'lstm.csv'
    tf_csv = tmp_path / 'tf.csv'
    xgb_csv = tmp_path / 'xgb.csv'
    _write_prob_csv(lstm_csv, 10, 0.40)
    _write_prob_csv(tf_csv, 10, 0.45)
    _write_prob_csv(xgb_csv, 10, 0.55)

    out_csv = tmp_path / 'pred.csv'
    args_inf = SimpleNamespace(
        meta_config=str(meta_path),
        base_probs=[f"lstm={lstm_csv}", f"tf={tf_csv}", f"xgb={xgb_csv}"],
        output=str(out_csv),
        threshold=None,
    )
    infer(args_inf)

    assert os.path.exists(str(out_csv))
    # check a few rows
    with open(out_csv, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert rows[0] == ['id','prob','decision']
    assert len(rows) == 11  # 10 samples + header

    # summary json
    summary = str(out_csv).replace('.csv', '.summary.json')
    assert os.path.exists(summary)
