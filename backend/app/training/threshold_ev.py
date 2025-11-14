import argparse
import csv
import json
import os
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import numpy as np

try:
    from ..db import engine
    from sqlmodel import Session, select
    from ..models import Trade
except Exception:
    engine = None  # type: ignore
    Session = None  # type: ignore
    Trade = None  # type: ignore


def _load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    probs: List[float] = []
    pnls: List[float] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                p = float(row.get("prob", row.get("probability", "")))
                pnl = float(row.get("pnl", row.get("pnl_pct", row.get("return", ""))))
                probs.append(p)
                pnls.append(pnl)
            except Exception:
                continue
    return np.asarray(probs, dtype=np.float32), np.asarray(pnls, dtype=np.float32)


def _load_json(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        data = json.load(f)
    probs: List[float] = []
    pnls: List[float] = []
    if isinstance(data, list):
        for obj in data:
            try:
                p = float(obj.get("prob") or obj.get("probability"))
                pnl = float(obj.get("pnl") or obj.get("pnl_pct") or obj.get("return"))
                probs.append(p)
                pnls.append(pnl)
            except Exception:
                continue
    elif isinstance(data, dict) and "records" in data:
        for obj in data["records"]:
            try:
                p = float(obj.get("prob") or obj.get("probability"))
                pnl = float(obj.get("pnl") or obj.get("pnl_pct") or obj.get("return"))
                probs.append(p)
                pnls.append(pnl)
            except Exception:
                continue
    return np.asarray(probs, dtype=np.float32), np.asarray(pnls, dtype=np.float32)


def _load_from_db(days: int = 30, symbol: str | None = None, exchange_type: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if Session is None or Trade is None or engine is None:  # type: ignore
        raise RuntimeError("DB access is unavailable; use --input CSV/JSON instead")
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))
    probs: List[float] = []
    pnls: List[float] = []
    with Session(engine) as s:  # type: ignore
        stmt = select(Trade).where(Trade.status == "closed")  # type: ignore
        if symbol:
            stmt = stmt.where(Trade.symbol == symbol)  # type: ignore
        if exchange_type:
            stmt = stmt.where(Trade.exchange_type == exchange_type)  # type: ignore
        stmt = stmt.where(Trade.closed_at != None)  # type: ignore
        rows = list(s.exec(stmt).all())
    for r in rows:
        try:
            if r.closed_at and r.closed_at.replace(tzinfo=timezone.utc) < cutoff:
                continue
            # Attempt to read probability used at entry from strategy_json if present
            p = None
            if r.strategy_json:
                try:
                    js = json.loads(r.strategy_json)
                    p = float(js.get("prob") or js.get("bottom_score"))
                except Exception:
                    p = None
            if p is None:
                continue
            # Prefer explicit realized PnL snapshot if present
            if r.pnl_pct_snapshot is not None:
                pnl = float(r.pnl_pct_snapshot)
            else:
                # Fallback: approximate from take_profit_pct/stop_loss_pct based on status; if unknown, skip
                pnl = None
                try:
                    if r.take_profit_pct and r.stop_loss_pct:
                        # no direction info of outcome; skip
                        pnl = None
                except Exception:
                    pnl = None
            if pnl is None:
                continue
            probs.append(float(p))
            pnls.append(float(pnl))
        except Exception:
            continue
    if not probs:
        raise RuntimeError("No usable trade records with (prob, pnl) found; provide --input CSV/JSON")
    return np.asarray(probs, dtype=np.float32), np.asarray(pnls, dtype=np.float32)


def optimize_ev(probs: np.ndarray, pnls: np.ndarray, t_low: float = 0.5, t_high: float = 0.995, steps: int = 60,
                min_coverage: float = 0.005, min_trades: int = 5) -> dict:
    assert probs.shape[0] == pnls.shape[0] and probs.ndim == 1
    thresholds = np.linspace(t_low, t_high, steps)
    n = len(probs)
    best = {"t": float(t_low), "ev": float("-inf"), "coverage": 0.0, "n": 0}
    for t in thresholds:
        mask = probs >= t
        k = int(mask.sum())
        if k <= 0:
            continue
        coverage = k / max(1, n)
        if coverage < min_coverage or k < min_trades:
            continue
        ev = float(np.mean(pnls[mask]))
        if ev > best["ev"]:
            best = {"t": float(t), "ev": ev, "coverage": float(coverage), "n": int(k)}
    return best


def _write_to_meta(meta_path: str, ev_info: dict) -> None:
    metrics_path = os.path.splitext(meta_path)[0] + ".metrics.json"
    payload = {}
    try:
        with open(metrics_path, "r") as f:
            payload = json.load(f)
    except Exception:
        payload = {}
    payload = payload or {}
    payload.setdefault("ev_opt", {})
    payload["ev_opt"].update({
        "best_threshold_ev": ev_info.get("t"),
        "ev_at_best": ev_info.get("ev"),
        "coverage_at_best": ev_info.get("coverage"),
        "n_at_best": ev_info.get("n"),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    try:
        with open(metrics_path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"EV optimum appended -> {metrics_path}")
    except Exception as e:
        print(f"Failed to write metrics sidecar: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="csv", choices=["csv", "json", "db"], help="Input source type")
    p.add_argument("--input", type=str, default="", help="Path to CSV/JSON with columns prob,pnl (ignored for db)")
    p.add_argument("--days", type=int, default=30, help="Lookback days for --source db")
    p.add_argument("--symbol", type=str, default="", help="Filter symbol for --source db")
    p.add_argument("--exchange-type", type=str, default="", help="Filter exchange_type for --source db")
    p.add_argument("--t-low", type=float, default=0.50)
    p.add_argument("--t-high", type=float, default=0.995)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--min-coverage", type=float, default=0.005)
    p.add_argument("--min-trades", type=int, default=5)
    p.add_argument("--write-to-meta", type=str, default="", help="Path to stacking_meta.json to append EV optimum into its metrics sidecar")
    args = p.parse_args()

    if args.source in ("csv", "json") and not args.input:
        raise SystemExit("--input file is required for source csv/json")

    if args.source == "csv":
        probs, pnls = _load_csv(args.input)
    elif args.source == "json":
        probs, pnls = _load_json(args.input)
    else:
        probs, pnls = _load_from_db(days=args.days, symbol=(args.symbol or None), exchange_type=(args.exchange_type or None))

    if len(probs) == 0:
        raise SystemExit("No records to evaluate")

    ev_best = optimize_ev(probs, pnls, t_low=args.t_low, t_high=args.t_high, steps=args.steps,
                          min_coverage=args.min_coverage, min_trades=args.min_trades)
    result = {
        "n_total": int(len(probs)),
        "t_low": float(args.t_low),
        "t_high": float(args.t_high),
        "steps": int(args.steps),
        "min_coverage": float(args.min_coverage),
        "min_trades": int(args.min_trades),
        "best": ev_best,
    }
    print(json.dumps(result, ensure_ascii=False))
    if args.write_to_meta:
        _write_to_meta(args.write_to_meta, ev_best)


if __name__ == "__main__":
    main()
