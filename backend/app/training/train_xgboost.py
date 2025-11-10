import argparse
import logging
import os
import pickle
import sys

import numpy as np

try:
    import xgboost as xgb
except Exception as e:
    raise RuntimeError("xgboost가 설치되어 있지 않습니다. requirements.txt에 xgboost>=2.0.0 추가 후 설치하세요.") from e

try:
    # Prefer absolute import when executed as a script
    from backend.app.training.dataset import build_tabular_dataset, train_val_split_time_order, build_bottom_tabular_dataset
except Exception:
    # Fallback: add repo root and retry
    sys.path.append('.')
    from backend.app.training.dataset import build_tabular_dataset, train_val_split_time_order, build_bottom_tabular_dataset

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
log = logging.getLogger("train_xgb")


def train(days: int, val_ratio: float, max_depth: int, n_estimators: int, learning_rate: float, model_out: str, mode: str, interval: str,
          past_window: int = 15, future_window: int = 60, min_gap: int = 20, tolerance_pct: float = 0.004, scale_pos_weight: float = -1.0):
    if mode == 'reg_next_ret':
        X, y, feature_names = build_tabular_dataset(days=days, interval=interval)
        objective = 'reg:squarederror'
        eval_metric = 'rmse'
    elif mode == 'cls_bottom':
        X, y, feature_names = build_bottom_tabular_dataset(days=days, interval=interval,
                                                           past_window=past_window, future_window=future_window,
                                                           min_gap=min_gap, tolerance_pct=tolerance_pct)
        objective = 'binary:logistic'
        eval_metric = 'logloss'
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    X_tr, X_va, y_tr, y_va = train_val_split_time_order(X, y, val_ratio=val_ratio)

    # Dataset usage summary logging (source already logged inside builders)
    total_rows = X.shape[0]
    if mode == 'cls_bottom':
        pos_total = int((y == 1).sum()); neg_total = int((y == 0).sum())
        pos_tr = int((y_tr == 1).sum()); neg_tr = int((y_tr == 0).sum())
        pos_va = int((y_va == 1).sum()); neg_va = int((y_va == 0).sum())
        log.info(
            "[dataset][split] mode=%s interval=%s total_rows=%d train_rows=%d val_rows=%d class_counts_total(pos=%d,neg=%d) train(pos=%d,neg=%d) val(pos=%d,neg=%d)",
            mode, interval, total_rows, X_tr.shape[0], X_va.shape[0], pos_total, neg_total, pos_tr, neg_tr, pos_va, neg_va
        )
    else:
        log.info(
            "[dataset][split] mode=%s interval=%s total_rows=%d train_rows=%d val_rows=%d target_stats(mean=%.6f,std=%.6f)",
            mode, interval, total_rows, X_tr.shape[0], X_va.shape[0], float(y_tr.mean()), float(y_tr.std())
        )

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dvalid = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)

    params = {
        'objective': objective,
        'max_depth': max_depth,
        'eta': learning_rate,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': eval_metric
    }
    # Handle class imbalance for classification
    if mode == 'cls_bottom':
        if scale_pos_weight is not None and scale_pos_weight < 0:
            # auto compute from training split
            pos = float((y_tr == 1).sum())
            neg = float((y_tr == 0).sum())
            if pos > 0:
                params['scale_pos_weight'] = max(1.0, neg / pos)
            else:
                params['scale_pos_weight'] = 1.0
        elif scale_pos_weight and scale_pos_weight > 0:
            params['scale_pos_weight'] = float(scale_pos_weight)
    evals = [(dtrain, 'train'), (dvalid, 'valid')]
    if mode == 'cls_bottom':
        pos_cnt = int((y_tr == 1).sum()); neg_cnt = int((y_tr == 0).sum())
        log.info("Training XGBoost: mode=%s interval=%s X_tr=%s X_va=%s features=%d class_counts(pos=%d,neg=%d) scale_pos_weight=%s",
                 mode, interval, X_tr.shape, X_va.shape, len(feature_names), pos_cnt, neg_cnt, str(params.get('scale_pos_weight', 'None')))
    else:
        log.info("Training XGBoost: mode=%s interval=%s X_tr=%s X_va=%s features=%d", mode, interval, X_tr.shape, X_va.shape, len(feature_names))
    booster = xgb.train(params, dtrain, num_boost_round=n_estimators, evals=evals, verbose_eval=False)

    # Evaluation
    pred = booster.predict(dvalid)
    metrics = {}
    if mode == 'reg_next_ret':
        rmse = float(np.sqrt(np.mean((pred - y_va) ** 2)))
        mae = float(np.mean(np.abs(pred - y_va)))
        direction_acc = float(np.mean(np.sign(pred) == np.sign(y_va)))
        metrics.update({'rmse': rmse, 'mae': mae, 'direction_acc': direction_acc})
        log.info("[XGB][reg] RMSE=%.6f MAE=%.6f DirAcc=%.3f", rmse, mae, direction_acc)
    else:
        cls = (pred >= 0.5).astype(np.int32)
        tp = int(((cls == 1) & (y_va == 1)).sum())
        fp = int(((cls == 1) & (y_va == 0)).sum())
        fn = int(((cls == 0) & (y_va == 1)).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        # threshold sweep for best F1
        best = {'threshold': 0.5, 'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}
        for th in np.linspace(0.05, 0.95, 19):
            c2 = (pred >= th).astype(np.int32)
            tp2 = int(((c2 == 1) & (y_va == 1)).sum())
            fp2 = int(((c2 == 1) & (y_va == 0)).sum())
            fn2 = int(((c2 == 0) & (y_va == 1)).sum())
            p2 = tp2 / (tp2 + fp2) if tp2 + fp2 > 0 else 0.0
            r2 = tp2 / (tp2 + fn2) if tp2 + fn2 > 0 else 0.0
            f12 = 2 * p2 * r2 / (p2 + r2) if p2 + r2 > 0 else 0.0
            if f12 > best['f1']:
                best = {'threshold': float(th), 'precision': float(p2), 'recall': float(r2), 'f1': float(f12), 'tp': tp2, 'fp': fp2, 'fn': fn2}
        metrics.update({'default_threshold': 0.5, 'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn,
                        'best_threshold': best['threshold'], 'best_precision': best['precision'], 'best_recall': best['recall'], 'best_f1': best['f1'],
                        'best_tp': best['tp'], 'best_fp': best['fp'], 'best_fn': best['fn']})
        log.info("[XGB][cls] @0.50 TP=%d FP=%d FN=%d P=%.4f R=%.4f F1=%.4f | best_th=%.2f F1=%.4f P=%.4f R=%.4f",
                 tp, fp, fn, precision, recall, f1, best['threshold'], best['f1'], best['precision'], best['recall'])

    out_dir = os.path.dirname(model_out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(model_out, 'wb') as f:
        pickle.dump({'booster': booster, 'features': feature_names, 'metrics': metrics, 'mode': mode}, f)
    # Persist metrics as JSON sidecar
    try:
        import json, time
        sidecar = os.path.splitext(model_out)[0] + '.metrics.json'
        record = {
            'model_out': model_out,
            'mode': mode,
            'interval': interval,
            'n_features': len(feature_names),
            'train_rows': int(dtrain.num_row()),
            'val_rows': int(dvalid.num_row()),
            'metrics': metrics,
            'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        with open(sidecar, 'w') as mf:
            json.dump(record, mf, ensure_ascii=False, indent=2)
        log.info("Metrics saved to %s", sidecar)
    except Exception as e:
        log.warning("Failed to write metrics sidecar: %s", e)
    log.info("Model saved to %s", model_out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--days', type=int, default=14)
    p.add_argument('--val-ratio', type=float, default=0.2)
    p.add_argument('--max-depth', type=int, default=6)
    p.add_argument('--n-estimators', type=int, default=300)
    p.add_argument('--learning-rate', type=float, default=0.05)
    p.add_argument('--model-out', type=str, default='backend/app/training/models/xgb_model.pkl')
    p.add_argument('--mode', type=str, default='reg_next_ret', choices=['reg_next_ret','cls_bottom'])
    p.add_argument('--interval', type=str, default='1m')
    # bottom labeling params
    p.add_argument('--past-window', type=int, default=15)
    p.add_argument('--future-window', type=int, default=60)
    p.add_argument('--min-gap', type=int, default=20)
    p.add_argument('--tolerance-pct', type=float, default=0.004)
    p.add_argument('--scale-pos-weight', type=float, default=-1.0, help='XGBoost scale_pos_weight; -1 to auto from train set')
    args = p.parse_args()
    train(days=args.days, val_ratio=args.val_ratio, max_depth=args.max_depth, n_estimators=args.n_estimators,
        learning_rate=args.learning_rate, model_out=args.model_out, mode=args.mode, interval=args.interval,
        past_window=args.past_window, future_window=args.future_window, min_gap=args.min_gap, tolerance_pct=args.tolerance_pct,
        scale_pos_weight=args.scale_pos_weight)


if __name__ == '__main__':
    main()
import argparse
import logging
import os
import pickle
import sys

import numpy as np
import xgboost as xgb

try:
    from .dataset import build_tabular_dataset, train_val_split_time_order, build_bottom_tabular_dataset
except Exception:
    # Allow running as a script from repo root
    sys.path.append('.')
    from backend.app.training.dataset import build_tabular_dataset, train_val_split_time_order, build_bottom_tabular_dataset

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
log = logging.getLogger("train_xgb")


def train(days: int, val_ratio: float, max_depth: int, n_estimators: int, learning_rate: float, model_out: str, mode: str, interval: str,
          past_window: int = 15, future_window: int = 60, min_gap: int = 20, tolerance_pct: float = 0.004, scale_pos_weight: float = -1.0):
    if mode == 'reg_next_ret':
        X, y, feature_names = build_tabular_dataset(days=days, interval=interval)
        objective = 'reg:squarederror'
        eval_metric = 'rmse'
    elif mode == 'cls_bottom':
        X, y, feature_names = build_bottom_tabular_dataset(days=days, interval=interval,
                                                           past_window=past_window, future_window=future_window,
                                                           min_gap=min_gap, tolerance_pct=tolerance_pct)
        objective = 'binary:logistic'
        eval_metric = 'logloss'
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    X_tr, X_va, y_tr, y_va = train_val_split_time_order(X, y, val_ratio=val_ratio)

    # Dataset usage summary logging (source already logged inside builders)
    total_rows = X.shape[0]
    if mode == 'cls_bottom':
        pos_total = int((y == 1).sum()); neg_total = int((y == 0).sum())
        pos_tr = int((y_tr == 1).sum()); neg_tr = int((y_tr == 0).sum())
        pos_va = int((y_va == 1).sum()); neg_va = int((y_va == 0).sum())
        log.info(
            "[dataset][split] mode=%s interval=%s total_rows=%d train_rows=%d val_rows=%d class_counts_total(pos=%d,neg=%d) train(pos=%d,neg=%d) val(pos=%d,neg=%d)",
            mode, interval, total_rows, X_tr.shape[0], X_va.shape[0], pos_total, neg_total, pos_tr, neg_tr, pos_va, neg_va
        )
    else:
        log.info(
            "[dataset][split] mode=%s interval=%s total_rows=%d train_rows=%d val_rows=%d target_stats(mean=%.6f,std=%.6f)",
            mode, interval, total_rows, X_tr.shape[0], X_va.shape[0], float(y_tr.mean()), float(y_tr.std())
        )

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dvalid = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)

    params = {
        'objective': objective,
        'max_depth': max_depth,
        'eta': learning_rate,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': eval_metric
    }
    # Handle class imbalance for classification
    if mode == 'cls_bottom':
        if scale_pos_weight is not None and scale_pos_weight < 0:
            # auto compute from training split
            pos = float((y_tr == 1).sum())
            neg = float((y_tr == 0).sum())
            if pos > 0:
                params['scale_pos_weight'] = max(1.0, neg / pos)
            else:
                params['scale_pos_weight'] = 1.0
        elif scale_pos_weight and scale_pos_weight > 0:
            params['scale_pos_weight'] = float(scale_pos_weight)
    evals = [(dtrain, 'train'), (dvalid, 'valid')]
    if mode == 'cls_bottom':
        pos_cnt = int((y_tr == 1).sum()); neg_cnt = int((y_tr == 0).sum())
        log.info("Training XGBoost: mode=%s interval=%s X_tr=%s X_va=%s features=%d class_counts(pos=%d,neg=%d) scale_pos_weight=%s",
                 mode, interval, X_tr.shape, X_va.shape, len(feature_names), pos_cnt, neg_cnt, str(params.get('scale_pos_weight', 'None')))
    else:
        log.info("Training XGBoost: mode=%s interval=%s X_tr=%s X_va=%s features=%d", mode, interval, X_tr.shape, X_va.shape, len(feature_names))
    booster = xgb.train(params, dtrain, num_boost_round=n_estimators, evals=evals, verbose_eval=False)

    # Evaluation
    pred = booster.predict(dvalid)
    metrics = {}
    if mode == 'reg_next_ret':
        rmse = float(np.sqrt(np.mean((pred - y_va) ** 2)))
        mae = float(np.mean(np.abs(pred - y_va)))
        direction_acc = float(np.mean(np.sign(pred) == np.sign(y_va)))
        metrics.update({'rmse': rmse, 'mae': mae, 'direction_acc': direction_acc})
        log.info("[XGB][reg] RMSE=%.6f MAE=%.6f DirAcc=%.3f", rmse, mae, direction_acc)
    else:
        cls = (pred >= 0.5).astype(np.int32)
        tp = int(((cls == 1) & (y_va == 1)).sum())
        fp = int(((cls == 1) & (y_va == 0)).sum())
        fn = int(((cls == 0) & (y_va == 1)).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        metrics.update({'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn})
        log.info("[XGB][cls] TP=%d FP=%d FN=%d Precision=%.4f Recall=%.4f F1=%.4f", tp, fp, fn, precision, recall, f1)

    out_dir = os.path.dirname(model_out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(model_out, 'wb') as f:
        pickle.dump({'booster': booster, 'features': feature_names, 'metrics': metrics, 'mode': mode}, f)
    log.info("Model saved to %s", model_out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--days', type=int, default=14)
    p.add_argument('--val-ratio', type=float, default=0.2)
    p.add_argument('--max-depth', type=int, default=6)
    p.add_argument('--n-estimators', type=int, default=300)
    p.add_argument('--learning-rate', type=float, default=0.05)
    p.add_argument('--model-out', type=str, default='backend/app/training/models/xgb_model.pkl')
    p.add_argument('--mode', type=str, default='reg_next_ret', choices=['reg_next_ret','cls_bottom'])
    p.add_argument('--interval', type=str, default='1m')
    # bottom labeling params
    p.add_argument('--past-window', type=int, default=15)
    p.add_argument('--future-window', type=int, default=60)
    p.add_argument('--min-gap', type=int, default=20)
    p.add_argument('--tolerance-pct', type=float, default=0.004)
    p.add_argument('--scale-pos-weight', type=float, default=-1.0, help='XGBoost scale_pos_weight; -1 to auto from train set')
    args = p.parse_args()
    train(days=args.days, val_ratio=args.val_ratio, max_depth=args.max_depth, n_estimators=args.n_estimators,
        learning_rate=args.learning_rate, model_out=args.model_out, mode=args.mode, interval=args.interval,
        past_window=args.past_window, future_window=args.future_window, min_gap=args.min_gap, tolerance_pct=args.tolerance_pct,
        scale_pos_weight=args.scale_pos_weight)


if __name__ == '__main__':
    main()
