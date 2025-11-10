import os
from backend.app.training.train_xgboost import train

def test_train_xgb_minimal(tmp_path):
    out_path = tmp_path / 'xgb_model_test.pkl'
    # Use small parameters for speed
    train(days=7, val_ratio=0.2, max_depth=3, n_estimators=10, learning_rate=0.1,
          model_out=str(out_path), mode='cls_bottom', interval='1m',
          past_window=15, future_window=60, min_gap=20, tolerance_pct=0.004,
          scale_pos_weight=-1.0, early_stopping_rounds=5, seed=42, subsample=0.8,
          colsample_bytree=0.8, min_child_weight=1.0)
    assert os.path.exists(str(out_path))
    sidecar = str(out_path).replace('.pkl', '.metrics.json')
    assert os.path.exists(sidecar)
