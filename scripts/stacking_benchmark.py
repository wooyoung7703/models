import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

"""Benchmark multiple stacking ensemble methods.

Runs backend.app.training.stacking for each requested ensemble method with identical
base model parameters, then loads the generated metrics sidecars and produces a
summary ranking JSON.

Usage (bash):
  python scripts/stacking_benchmark.py \
    --methods logistic bayes dynamic lgbm \
    --days 30 --seq-len 30 --val-ratio 0.2 \
    --models lstm tf xgb --data-source real \
    --output scripts/stacking_benchmark_result.json

Light run (fast):
  python scripts/stacking_benchmark.py --methods logistic bayes --days 14 --quick --output scripts/stacking_benchmark_quick.json

Notes:
  - Requires dependencies for chosen methods (scikit-learn, xgboost, lightgbm).
  - dynamic uses recent-frac=0.2 (default). Adjust via --recent-frac.
  - lgbm will be slower; include only if you need non-linear baseline.
  - Each method writes its own stacking_meta_*METHOD*.json under models dir.
"""

import argparse


def run_method(args, method: str, models_dir: Path) -> Dict[str, float]:
    meta_out = models_dir / f"stacking_meta_{method}.json"
    cmd = [
        sys.executable,
        "-m",
        "backend.app.training.stacking",
        "--ensemble", method,
        "--meta-out", str(meta_out),
        "--val-ratio", str(args.val_ratio),
        "--seq-len", str(args.seq_len),
        "--days", str(args.days),
        "--past-window", str(args.past_window),
        "--future-window", str(args.future_window),
        "--min-gap", str(args.min_gap),
        "--tolerance-pct", str(args.tolerance_pct),
        "--data-source", args.data_source,
        "--models",
    ] + args.models
    if args.quick:
        cmd.append("--quick-refit")
    if args.use_ordinal:
        cmd.append("--use-ordinal")
    if method == "bayes":
        cmd += ["--bayes-alpha", str(args.bayes_alpha)]
    if method == "dynamic":
        cmd += ["--recent-frac", str(args.recent_frac)]
    if args.bagging_n > 1:
        cmd += ["--bagging-n", str(args.bagging_n)]
    if args.oof_folds > 1:
        cmd += ["--oof-folds", str(args.oof_folds)]

    start = time.time()
    print(f"[benchmark] Running {method} -> {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dur = time.time() - start
    if proc.returncode != 0:
        print(f"[benchmark] Method {method} failed (rc={proc.returncode})")
        print(proc.stdout)
        print(proc.stderr)
        return {"method": method, "error": proc.stderr.strip(), "duration_sec": dur}
    metrics_path = meta_out.with_suffix(".metrics.json")
    if not metrics_path.exists():
        return {"method": method, "error": "metrics_sidecar_missing", "duration_sec": dur}
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    # Extract key metrics safely
    result = {
        "method": method,
        "duration_sec": dur,
        "best_threshold_precision": data.get("best_threshold_precision"),
        "precision_at_best_t": data.get("precision_at_best_t"),
        "coverage_at_best_t": data.get("coverage_at_best_t"),
        "precision_high": data.get("precision_high"),
        "precision_low": data.get("precision_low"),
    }
    # Add classification report metrics if present
    metrics = data.get("metrics", {})
    for k in ["precision", "recall", "f1", "support"]:
        v = metrics.get(k)
        if isinstance(v, (int, float)):
            result[f"metrics_{k}"] = v
    prk = data.get("metrics_prk", {})
    for k in ["precision_top1pct", "recall_top1pct", "tp_top1pct", "fp_top1pct"]:
        v = prk.get(k)
        if isinstance(v, (int, float)):
            result[f"prk_{k}"] = v
    return result


def rank_results(results: List[Dict[str, float]]) -> List[Dict[str, float]]:
    # Primary sort: precision_at_best_t, secondary: coverage_at_best_t
    def key(r):
        return (
            -(r.get("precision_at_best_t") or 0.0),
            -(r.get("coverage_at_best_t") or 0.0),
        )
    return sorted(results, key=key)


def main():
    p = argparse.ArgumentParser(description="Benchmark stacking ensemble methods")
    p.add_argument("--methods", nargs="+", default=["logistic", "bayes"], help="Ensemble methods to test")
    p.add_argument("--models", nargs="+", default=["lstm", "tf", "xgb"], help="Base models list")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--past-window", type=int, default=15)
    p.add_argument("--future-window", type=int, default=60)
    p.add_argument("--min-gap", type=int, default=20)
    p.add_argument("--tolerance-pct", type=float, default=0.004)
    p.add_argument("--data-source", type=str, default="real", choices=["real", "synthetic"])
    p.add_argument("--bayes-alpha", type=float, default=1.0)
    p.add_argument("--recent-frac", type=float, default=0.2)
    p.add_argument("--bagging-n", type=int, default=1)
    p.add_argument("--oof-folds", type=int, default=0)
    p.add_argument("--use-ordinal", action="store_true", help="Include ordinal LSTM signals")
    p.add_argument("--quick", action="store_true", help="Enable quick-refit mode for fast calibration runs")
    p.add_argument("--output", type=str, default="scripts/stacking_benchmark_result.json")
    args = p.parse_args()

    models_dir = Path("backend/app/training/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, float]] = []
    for m in args.methods:
        res = run_method(args, m, models_dir)
        results.append(res)
    ranked = rank_results([r for r in results if "error" not in r])
    summary = {
        "methods_requested": args.methods,
        "results": results,
        "ranked": ranked,
        "primary_sort": "precision_at_best_t desc, coverage_at_best_t desc",
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[benchmark] Summary written -> {out_path}")
    # Print top line for quick view
    if ranked:
        top = ranked[0]
        print("[benchmark] TOP method:", top.get("method"), "precision_at_best_t=", top.get("precision_at_best_t"))


if __name__ == "__main__":
    main()
