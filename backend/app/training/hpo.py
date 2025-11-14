import argparse
import json
import os
import tempfile
import sys
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import optuna

DEFAULT_BOTTOM_TRAIN_DAYS = int(os.getenv("BOTTOM_TRAIN_DAYS", "14"))
try:
    from backend.app.core.config import settings
except Exception:
    if '.' not in sys.path:
        sys.path.append('.')
    try:
        from backend.app.core.config import settings  # type: ignore
    except Exception:
        settings = None  # type: ignore
if 'settings' in locals() and settings is not None:
    DEFAULT_BOTTOM_TRAIN_DAYS = getattr(settings, 'BOTTOM_TRAIN_DAYS', DEFAULT_BOTTOM_TRAIN_DAYS)

# Training imports (reuse existing scripts)
from backend.app.training.train_lstm import train as train_lstm
from backend.app.training.train_transformer import train as train_transformer
from backend.app.training.train_xgboost import train as train_xgb


def _ensure_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        'batch_size': 32,
        'num_workers': 0,
        'scheduler': 'none',
        'grad_clip': 0.0,
        'patience': 5,
        'early_stop_metric': 'f1' if args.mode == 'cls_bottom' else 'rmse',
        'amp': False,
        'pos_weight': -1.0,
        'calibration': 'none',
        'min_coverage': 0.005,
        'regime_filter': 'none',
        'regime_percentile': 0.5,
        't_low': 0.50,
        't_high': 0.995,
        'hidden_dim': 32,  # placeholders; will be overwritten by trial for lstm
        'model_dim': 32,   # placeholders; will be overwritten by trial for transformer
        'nhead': 4,
        'dropout': 0.0,
        'num_layers': 1,
        'device': getattr(args, 'device', 'cpu'),
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


def _objective_lstm(trial: optuna.Trial, base_args: argparse.Namespace) -> float:
    base_args = _ensure_defaults(base_args)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64])
    num_layers = trial.suggest_int('num_layers', 1, 2)
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    # epochs kept low for speed; use base_args.epochs if provided
    epochs = getattr(base_args, 'epochs', 2)
    mode = base_args.mode
    model_out = os.path.join(base_args.output_dir, f"trial_lstm_{trial.number}.pt")
    args = argparse.Namespace(**{**vars(base_args), **{
        'lr': lr,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'epochs': epochs,
        'model_out': model_out,
    }})
    try:
        train_lstm(args)
    except Exception:
        return 0.0
    sidecar = os.path.splitext(model_out)[0] + '.metrics.json'
    metric = _extract_metric(sidecar, mode, trial, base_args.metric)
    return metric


def _objective_transformer(trial: optuna.Trial, base_args: argparse.Namespace) -> float:
    base_args = _ensure_defaults(base_args)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    model_dim = trial.suggest_categorical('model_dim', [32, 64])
    num_layers = trial.suggest_int('num_layers', 1, 2)
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    epochs = getattr(base_args, 'epochs', 2)
    mode = base_args.mode
    model_out = os.path.join(base_args.output_dir, f"trial_tf_{trial.number}.pt")
    args = argparse.Namespace(**{**vars(base_args), **{
        'lr': lr,
        'model_dim': model_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'epochs': epochs,
        'model_out': model_out,
    }})
    try:
        train_transformer(args)
    except Exception:
        return 0.0
    sidecar = os.path.splitext(model_out)[0] + '.metrics.json'
    metric = _extract_metric(sidecar, mode, trial, base_args.metric)
    return metric


def _objective_xgb(trial: optuna.Trial, base_args: argparse.Namespace) -> float:
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 6)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
    n_estimators = trial.suggest_int('n_estimators', 80, 200)
    model_out = os.path.join(base_args.output_dir, f"trial_xgb_{trial.number}.pkl")
    args_dict = {
        'days': base_args.days,
        'val_ratio': base_args.val_ratio,
        'max_depth': max_depth,
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'model_out': model_out,
        'mode': base_args.mode,
        'interval': base_args.interval,
        'past_window': base_args.past_window,
        'future_window': base_args.future_window,
        'min_gap': base_args.min_gap,
        'tolerance_pct': base_args.tolerance_pct,
        'scale_pos_weight': -1.0,
        'early_stopping_rounds': 30,
        'seed': base_args.seed,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'min_child_weight': 1.0,
    }
    try:
        train_xgb(**args_dict)
    except Exception:
        return 0.0 if base_args.mode == 'cls_bottom' else float('inf')
    sidecar = os.path.splitext(model_out)[0] + '.metrics.json'
    metric = _extract_metric(sidecar, base_args.mode, trial, base_args.metric)
    return metric


def _extract_metric(sidecar: str, mode: str, trial: optuna.Trial, metric_name: str) -> float:
    if not os.path.exists(sidecar):
        # Penalize missing sidecar
        return 0.0 if 'cls' in mode else float('inf')
    try:
        with open(sidecar, 'r') as f:
            data = json.load(f)
    except Exception:
        return 0.0 if 'cls' in mode else float('inf')
    metrics = data.get('metrics', {}) if 'metrics' in data else data.get('metrics', {})
    # classification maximize, regression minimize
    if mode == 'cls_bottom':
        # default fallback f1
        val = float(metrics.get(metric_name, metrics.get('f1', 0.0)))
        return val
    else:
        # regression: rmse
        val = float(metrics.get(metric_name, metrics.get('rmse', float('inf'))))
        # Optuna always maximizes by default if direction='maximize'; handle externally
        return val


def run_study(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.output_dir, exist_ok=True)
    storage = None  # in-memory
    study_name = f"hpo_{args.model}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    direction = 'maximize' if args.mode == 'cls_bottom' else 'minimize'
    study = optuna.create_study(direction=direction, study_name=study_name, storage=storage)

    # Base args passed to objective
    base_args = args

    if args.model == 'lstm':
        obj = lambda trial: _objective_lstm(trial, base_args)
    elif args.model == 'transformer':
        obj = lambda trial: _objective_transformer(trial, base_args)
    elif args.model == 'xgb':
        obj = lambda trial: _objective_xgb(trial, base_args)
    else:
        raise ValueError('Unsupported model type')

    study.optimize(obj, n_trials=args.trials, timeout=args.timeout)

    # Handle case of zero completed trials gracefully
    if all(t.state != optuna.trial.TrialState.COMPLETE for t in study.trials):
        result = {
            'study_name': study_name,
            'direction': direction,
            'best_value': None,
            'best_params': {},
            'metric': args.metric,
            'model': args.model,
            'mode': args.mode,
            'trials': len(study.trials),
            'status': 'no_completed_trials'
        }
        with open(os.path.join(args.output_dir, 'best_params.json'), 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    best = study.best_trial
    result = {
        'study_name': study_name,
        'direction': direction,
        'best_value': best.value,
        'best_params': best.params,
        'metric': args.metric,
        'model': args.model,
        'mode': args.mode,
        'trials': len(study.trials),
    }
    with open(os.path.join(args.output_dir, 'best_params.json'), 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main():
    p = argparse.ArgumentParser(description='Optuna HPO runner for LSTM/Transformer/XGBoost')
    p.add_argument('--model', type=str, required=True, choices=['lstm','transformer','xgb'])
    p.add_argument('--mode', type=str, default='cls_bottom', choices=['cls_bottom','reg_next_ret'])
    p.add_argument('--interval', type=str, default='1m')
    p.add_argument('--days', type=int, default=DEFAULT_BOTTOM_TRAIN_DAYS)
    p.add_argument('--seq-len', type=int, default=30)
    p.add_argument('--val-ratio', type=float, default=0.2)
    p.add_argument('--trials', type=int, default=10)
    p.add_argument('--timeout', type=int, default=0, help='Global timeout seconds (0 for no timeout)')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--metric', type=str, default='f1', help='Metric key to optimize (classification: f1/precision_high etc, regression: rmse)')
    p.add_argument('--output-dir', type=str, default='backend/app/training/hpo_runs')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cpu')
    # bottom labeling settings
    p.add_argument('--past-window', type=int, default=15)
    p.add_argument('--future-window', type=int, default=60)
    p.add_argument('--min-gap', type=int, default=20)
    p.add_argument('--tolerance-pct', type=float, default=0.004)
    args = p.parse_args()
    res = run_study(args)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    main()