import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Resolve repo root for running module commands reliably
try:
    from .core.config import settings
    from .model_adapters import registry
except Exception:
    # Allow running as a module from repo root
    if '.' not in sys.path:
        sys.path.append('.')
    from backend.app.core.config import settings  # type: ignore
    from backend.app.model_adapters import registry  # type: ignore

log = logging.getLogger("scheduler")

_scheduler: Optional[AsyncIOScheduler] = None
_training_active: bool = False
_last_retrain_utc: Optional[datetime] = None
_last_retrain_reason: Optional[str] = None
_last_retrain_base_days: Optional[int] = None
_last_retrain_stacking_days: Optional[int] = None
_last_retrain_type: Optional[str] = None  # monthly | trade-close | prob-drift | manual
_last_meta_retrain_utc: Optional[datetime] = None
_last_meta_retrain_reason: Optional[str] = None  # interval | daily | manual-meta
_last_meta_retrain_exit_code: Optional[int] = None
_last_meta_retrain_overwritten: Optional[bool] = None
_pending_meta_retrain: bool = False  # set if skipped due to active training
_meta_history: list[dict] = []  # recent meta retrain records (improvement etc)
# Derived stats for meta retrain history
_meta_corr_rel_improve_samples: Optional[float] = None  # Pearson correlation(rel_improve, samples)
_meta_neg_streak: int = 0  # consecutive negative/failed improvement streak
_meta_reg_slope: Optional[float] = None  # linear regression slope (rel_improve ~ samples)
_meta_reg_pvalue: Optional[float] = None  # two-tailed p-value for slope
_last_good_meta_snapshot: Optional[str] = None  # path to last good (overwritten & positive improve) meta snapshot

def _linear_regression(x: list[float], y: list[float]) -> tuple[Optional[float], Optional[float]]:
    """Return (slope, pvalue) for simple linear regression y ~ a + b*x.

    Uses standard formulas with t-statistic for slope significance.
    Returns (None, None) if insufficient data (<5) or zero variance in x.
    """
    import math
    n = len(x)
    if n < 5:
        return (None, None)
    mx = sum(x) / n
    my = sum(y) / n
    var_x = sum((xi - mx) ** 2 for xi in x)
    if var_x == 0:
        return (None, None)
    cov_xy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    slope = cov_xy / var_x
    # Residual variance
    intercept = my - slope * mx
    residuals = [yi - (intercept + slope * xi) for xi, yi in zip(x, y)]
    rss = sum(r * r for r in residuals)
    df = n - 2
    if df <= 0:
        return (slope, None)
    s2 = rss / df
    se_slope = math.sqrt(s2 / var_x) if var_x else None
    if not se_slope or se_slope == 0:
        return (slope, None)
    t_stat = slope / se_slope
    # Compute two-tailed p-value for t-statistic.
    df_float = float(df)
    use_normal = df > 40 or getattr(settings, 'META_REG_USE_NORMAL_APPROX', 1)
    if use_normal:
        # Normal approximation
        z = abs(t_stat)
        def _erf(v: float) -> float:
            a1, a2, a3, a4, a5, p = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
            sign = 1 if v >= 0 else -1
            v = abs(v)
            t = 1.0 / (1.0 + p * v)
            y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-v * v)
            return sign * y
        def _normal_cdf(v: float) -> float:
            return 0.5 * (1 + _erf(v / math.sqrt(2)))
        p_one_tail = 1 - _normal_cdf(z)
        pvalue = max(0.0, min(1.0, 2 * p_one_tail))
        return (slope, pvalue)
    # Student's t exact CDF via incomplete beta (if mpmath available), else fallback to normal.
    try:
        import mpmath as mp
        # CDF for t using regularized incomplete beta: F(t) = 0.5 + t * BetaInc(df/2, 0.5, df/(df+t^2)) / (2 * sqrt(df) * B(df/2, 0.5))
        # We'll use mp.qf for clarity; implement two-tailed p.
        x_val = abs(t_stat)
        # Regularized incomplete beta I_{v/(v+x^2)}(v/2, 1/2)
        v = df_float
        z_arg = v / (v + x_val * x_val)
        ib = mp.betainc(v/2.0, 0.5, 0, z_arg, regularized=True)
        # Derivation: two-tailed p = 2 * min(F(t), 1 - F(t))
        # F(t) formulation: F(t) = 1 - 0.5 * ib  (for t>0)
        F = 1 - 0.5 * ib
        p_two = 2 * min(F, 1 - F)
        p_two = float(max(0.0, min(1.0, p_two)))
        return (slope, p_two)
    except Exception:
        z = abs(t_stat)
        def _erf(v: float) -> float:
            a1, a2, a3, a4, a5, p = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
            sign = 1 if v >= 0 else -1
            v = abs(v)
            t = 1.0 / (1.0 + p * v)
            y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-v * v)
            return sign * y
        def _normal_cdf(v: float) -> float:
            return 0.5 * (1 + _erf(v / math.sqrt(2)))
        p_one_tail = 1 - _normal_cdf(z)
        pvalue = max(0.0, min(1.0, 2 * p_one_tail))
        return (slope, pvalue)

def _compute_meta_history_stats() -> None:
    """Recompute correlation and negative streak based on _meta_history.

    Negative record definition:
      - overwritten is False (no overwrite due to insufficient improvement) OR
      - rel_improve <= 0 (non-positive relative Brier improvement)
    Streak counts consecutive negative records from the tail.
    """
    global _meta_corr_rel_improve_samples, _meta_neg_streak
    # Correlation
    pairs: list[tuple[float, float]] = []
    for rec in _meta_history[-200:]:  # limit to recent window
        imp = rec.get('rel_improve')
        samp = rec.get('samples')
        if isinstance(imp, (int, float)) and isinstance(samp, (int, float)):
            pairs.append((float(imp), float(samp)))
    if len(pairs) >= 3:
        # Pearson correlation
        import math
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
        den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
        corr = num / (den_x * den_y) if den_x and den_y else None
        _meta_corr_rel_improve_samples = corr
    else:
        _meta_corr_rel_improve_samples = None
    # Negative streak (tail)
    streak = 0
    for rec in reversed(_meta_history):
        imp = rec.get('rel_improve')
        overwritten = rec.get('overwritten')
        negative = (overwritten is False) or (isinstance(imp, (int, float)) and imp <= 0.0)
        if negative:
            streak += 1
        else:
            break
    _meta_neg_streak = streak
    # Linear regression slope & p-value over rolling window
    global _meta_reg_slope, _meta_reg_pvalue
    try:
        window = int(getattr(settings, 'META_REG_WINDOW', 30))
    except Exception:
        window = 30
    subset = _meta_history[-window:]
    xs: list[float] = []
    ys: list[float] = []
    for rec in subset:
        imp = rec.get('rel_improve')
        samp = rec.get('samples')
        if isinstance(imp, (int, float)) and isinstance(samp, (int, float)):
            ys.append(float(imp))
            xs.append(float(samp))
    if len(xs) >= 5:
        slope, pval = _linear_regression(xs, ys)
        _meta_reg_slope, _meta_reg_pvalue = slope, pval
    else:
        _meta_reg_slope, _meta_reg_pvalue = None, None


def _python_executable() -> str:
    return sys.executable or "python"


async def _run_cmd_async(cmd: list[str], cwd: Optional[str] = None, env: Optional[dict] = None) -> int:
    """Run a subprocess command asynchronously and stream output to logs."""
    log.info("[job] run: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
        env=env or os.environ.copy(),
    )
    assert proc.stdout is not None
    # Stream output
    async for line in proc.stdout:
        try:
            log.info("[job] %s", line.decode(errors='ignore').rstrip())
        except Exception:
            pass
    rc = await proc.wait()
    log.info("[job] exit code: %s", rc)
    return int(rc)


async def run_training_sequence(base_days: int, stacking_days: int, reason: str) -> None:
    """Generic sequential training (XGBoost→LSTM→Transformer→Stacking) with configurable day windows.

    base_days: window for base models
    stacking_days: window for stacking meta
    reason: log annotation (e.g. 'monthly', 'trade-close')
    """
    start_ts = datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    log.info("[train-seq][%s] start=%s base_days=%d stacking_days=%d", reason, start_ts, base_days, stacking_days)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    py = _python_executable(); env = os.environ.copy()
    interval = getattr(settings, 'INTERVAL', '1m')
    seq_len = getattr(settings, 'SEQ_LEN', 30)

    xgb_out = getattr(settings, 'MODEL_XGB_PATH', 'backend/app/training/models/xgb_model.pkl')
    lstm_out = getattr(settings, 'MODEL_LSTM_PATH', 'backend/app/training/models/lstm_model.pt')
    tf_out = getattr(settings, 'MODEL_TRANSFORMER_PATH', 'backend/app/training/models/transformer_model.pt')
    stacking_out = getattr(settings, 'STACKING_META_PATH', 'backend/app/training/models/stacking_meta.json')

    cmd_xgb = [py, '-m', 'backend.app.training.train_xgboost', '--mode', 'cls_bottom', '--days', str(base_days), '--interval', interval,
               '--data-source', 'real', '--use-feature-set', '16', '--model-out', xgb_out, '--val-ratio', '0.2']
    cmd_lstm = [py, '-m', 'backend.app.training.train_lstm', '--mode', 'cls_bottom', '--days', str(base_days), '--seq-len', str(seq_len), '--interval', interval,
                '--data-source', 'real', '--use-feature-set', '16', '--model-out', lstm_out, '--val-ratio', '0.2']
    cmd_tf = [py, '-m', 'backend.app.training.train_transformer', '--mode', 'cls_bottom', '--days', str(base_days), '--seq-len', str(seq_len), '--interval', interval,
              '--data-source', 'real', '--use-feature-set', '16', '--model-out', tf_out, '--val-ratio', '0.2']
    cmd_stack = [py, '-m', 'backend.app.training.stacking', '--data-source', 'real', '--use-feature-set', '16', '--seq-len', str(seq_len), '--days', str(stacking_days),
                 '--interval', interval, '--models', 'lstm', 'tf', 'xgb', '--ensemble', 'logistic', '--meta-out', stacking_out]

    for name, cmd in (("xgboost", cmd_xgb), ("lstm", cmd_lstm), ("transformer", cmd_tf), ("stacking", cmd_stack)):
        rc = await _run_cmd_async(cmd, cwd=repo_root, env=env)
        if rc != 0:
            log.error("[train-seq][%s] %s failed rc=%d abort", reason, name, rc)
            return
    try:
        registry.load_from_settings()
        log.info("[train-seq][%s] registry reloaded: %s", reason, registry.status())
    except Exception as e:
        log.exception("[train-seq][%s] registry reload failed: %s", reason, e)
    log.info("[train-seq][%s] completed", reason)
    global _last_retrain_utc, _last_retrain_reason, _last_retrain_base_days, _last_retrain_stacking_days
    _last_retrain_utc = datetime.now(tz=timezone.utc)
    _last_retrain_reason = reason
    _last_retrain_base_days = base_days
    _last_retrain_stacking_days = stacking_days
    global _last_retrain_type
    _last_retrain_type = reason
    # If a meta retrain was pending (skipped earlier), schedule immediately after training
    try:
        global _pending_meta_retrain
        if _pending_meta_retrain and can_trigger_meta_retrain():
            loop = asyncio.get_event_loop()
            loop.create_task(run_meta_retrain_job())
            log.info('[train-seq][%s] dispatched pending meta-retrain', reason)
            _pending_meta_retrain = False
    except Exception:
        pass


def start_scheduler(timezone_name: Optional[str] = None) -> AsyncIOScheduler:
    """Start AsyncIO scheduler with a monthly cron job at 1st 00:00.

    timezone_name: e.g. 'UTC' or 'Asia/Seoul'. If None, APScheduler default is used.
    """
    global _scheduler
    if _scheduler:
        return _scheduler
    tz = None
    if timezone_name:
        try:
            from zoneinfo import ZoneInfo  # Python 3.9+
            tz = ZoneInfo(timezone_name)
        except Exception:
            tz = None
    _scheduler = AsyncIOScheduler(timezone=tz)
    trigger = CronTrigger(day=1, hour=0, minute=0, timezone=tz)
    _scheduler.add_job(run_monthly_training, trigger, id='monthly_train', name='Monthly Train 3 Models')
    # Meta retrain scheduling (bottom-vs-forecast logistic layer)
    try:
        if getattr(settings, 'META_RETRAIN_ENABLED', False):
            every_min = int(getattr(settings, 'META_RETRAIN_EVERY_MINUTES', 0) or 0)
            daily_at = str(getattr(settings, 'META_RETRAIN_DAILY_AT', '') or '').strip()
            if every_min > 0:
                _scheduler.add_job(run_meta_retrain_job, 'interval', minutes=every_min, id='meta_retrain_interval', max_instances=1, coalesce=True)
                log.info('[scheduler] meta retrain interval scheduled every %d min', every_min)
            elif daily_at:
                try:
                    hh, mm = daily_at.split(':')
                    _scheduler.add_job(run_meta_retrain_job, CronTrigger(hour=int(hh), minute=int(mm), timezone=tz), id='meta_retrain_daily', max_instances=1, coalesce=True)
                    log.info('[scheduler] meta retrain daily scheduled at %s', daily_at)
                except Exception:
                    log.error('[scheduler] invalid META_RETRAIN_DAILY_AT format (HH:MM): %s', daily_at)
            # Optional daily evaluation CSV generation slightly before daily meta retrain (fixed 00:02 UTC)
            try:
                _scheduler.add_job(run_eval_csv_job, CronTrigger(hour=0, minute=2, timezone=tz), id='bvf_eval_csv_daily', max_instances=1, coalesce=True)
                log.info('[scheduler] daily bvf eval CSV job scheduled 00:02')
            except Exception as _e:
                log.warning('[scheduler] failed to schedule eval CSV job: %s', _e)
    except Exception as e:
        log.exception('[scheduler] failed to schedule meta retrain jobs: %s', e)
    _scheduler.start()
    log.info("[scheduler] started; next run(s): %s", _scheduler.get_jobs())
    return _scheduler


def get_next_runs_summary() -> dict:
    out: dict = {}
    try:
        if _scheduler:
            for job in _scheduler.get_jobs():
                name = job.id
                next_run = job.next_run_time
                if next_run:
                    from datetime import datetime, timezone
                    now = datetime.now(tz=next_run.tzinfo or timezone.utc)
                    eta_seconds = int((next_run - now).total_seconds())
                    out[name] = {
                        'next_run_iso': next_run.isoformat(),
                        'eta_seconds': eta_seconds,
                    }
                else:
                    out[name] = {'next_run_iso': None, 'eta_seconds': None}
    except Exception as e:
        log.debug('[scheduler] get_next_runs_summary failed: %s', e)
    return out


async def run_monthly_training() -> None:
    """APScheduler job: run standard monthly training sequence.

    Uses BOTTOM_TRAIN_DAYS for base models and STACKING_META_DAYS for stacking meta.
    """
    base_days = getattr(settings, 'BOTTOM_TRAIN_DAYS', 30)
    stacking_days = getattr(settings, 'STACKING_META_DAYS', base_days)
    await run_training_sequence(int(base_days), int(stacking_days), reason='monthly')


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler:
        try:
            _scheduler.shutdown(wait=False)
            log.info("[scheduler] stopped")
        except Exception:
            pass
        _scheduler = None


async def trigger_now_background() -> None:
    """Helper to kick off the monthly sequence immediately (manual trigger)."""
    global _training_active
    if _training_active:
        log.info("[trigger] training already active; skipping manual trigger")
        return
    _training_active = True
    try:
        base_days = getattr(settings, 'BOTTOM_TRAIN_DAYS', 30)
        stacking_days = getattr(settings, 'STACKING_META_DAYS', base_days)
        await run_training_sequence(base_days, stacking_days, reason='manual')
    finally:
        _training_active = False


def can_trigger_trade_close() -> bool:
    if not getattr(settings, 'RETRAIN_ON_TRADE_CLOSE', False):
        return False
    global _last_retrain_utc, _training_active
    if _training_active:
        return False
    min_hours = float(getattr(settings, 'RETRAIN_MIN_INTERVAL_HOURS', 12.0))
    if _last_retrain_utc is None:
        return True
    diff = (datetime.now(tz=timezone.utc) - _last_retrain_utc).total_seconds() / 3600.0
    return diff >= min_hours


def trigger_trade_close_retrain() -> bool:
    """Schedule an immediate async retrain if allowed by settings & cooldown.

    Returns True if scheduled, False otherwise.
    """
    if not can_trigger_trade_close():
        return False
    base_days = getattr(settings, 'RETRAIN_DAYS', getattr(settings, 'BOTTOM_TRAIN_DAYS', 30))
    stacking_days = getattr(settings, 'RETRAIN_STACKING_DAYS', getattr(settings, 'STACKING_META_DAYS', base_days))
    async def _run():
        global _training_active
        if _training_active:
            return
        _training_active = True
        try:
            await run_training_sequence(base_days, stacking_days, reason='trade-close')
        finally:
            _training_active = False
    loop = asyncio.get_event_loop(); loop.create_task(_run())
    logging.getLogger("scheduler").info("[event] trade-close retrain scheduled base_days=%d stacking_days=%d", base_days, stacking_days)
    return True

def get_last_retrain_meta() -> dict:
    warn_thresh = int(getattr(settings, 'META_NEG_STREAK_WARN', 3))
    cos_warn = float(getattr(settings, 'META_COEF_DRIFT_WARN_COS', 0.92))
    # Attempt coefficient drift calc (cosine similarity current vs last good snapshot)
    coef_cos = None
    drift_warn = False
    try:
        meta_path = getattr(settings, 'BOTTOM_VS_FORECAST_META_PATH', 'data/bottom_vs_forecast_meta.json')
        last_good = _last_good_meta_snapshot or getattr(settings, 'META_LAST_GOOD_META_PATH', 'data/bottom_vs_forecast_meta_last_good.json')
        import json as _json, math
        if os.path.exists(meta_path) and os.path.exists(last_good):
            cur_obj = _json.loads(open(meta_path,'r',encoding='utf-8').read())
            good_obj = _json.loads(open(last_good,'r',encoding='utf-8').read())
            cur = cur_obj.get('coef') or []
            good = good_obj.get('coef') or []
            if isinstance(cur, list) and isinstance(good, list) and len(cur) == len(good) and len(cur) > 0:
                # cosine similarity
                dot = sum(float(a)*float(b) for a,b in zip(cur, good))
                na = math.sqrt(sum(float(a)*float(a) for a in cur))
                nb = math.sqrt(sum(float(b)*float(b) for b in good))
                if na and nb:
                    coef_cos = dot / (na*nb)
                    drift_warn = coef_cos < cos_warn
    except Exception as _e:
        pass
    return {
        'last_retrain_utc': _last_retrain_utc.isoformat() if _last_retrain_utc else None,
        'last_retrain_reason': _last_retrain_reason,
        'last_retrain_base_days': _last_retrain_base_days,
        'last_retrain_stacking_days': _last_retrain_stacking_days,
        'training_active': _training_active,
        'last_retrain_type': _last_retrain_type,
        'last_meta_retrain_utc': _last_meta_retrain_utc.isoformat() if _last_meta_retrain_utc else None,
        'last_meta_retrain_reason': _last_meta_retrain_reason,
        'last_meta_retrain_exit_code': _last_meta_retrain_exit_code,
        'last_meta_retrain_overwritten': _last_meta_retrain_overwritten,
        'pending_meta_retrain': _pending_meta_retrain,
        'meta_config': {
            'enabled': getattr(settings, 'META_RETRAIN_ENABLED', False),
            'every_minutes': getattr(settings, 'META_RETRAIN_EVERY_MINUTES', 0),
            'daily_at': getattr(settings, 'META_RETRAIN_DAILY_AT', ''),
            'min_interval_minutes': getattr(settings, 'META_RETRAIN_MIN_INTERVAL_MINUTES', 60),
            'min_rel_brier_improve': getattr(settings, 'META_RETRAIN_MIN_REL_BRIER_IMPROVE', 0.0),
        },
        'next_runs': get_next_runs_summary(),
        'meta_history': list(_meta_history[-40:]),
        'meta_corr_rel_improve_samples': _meta_corr_rel_improve_samples,
        'meta_neg_streak': _meta_neg_streak,
        'meta_neg_streak_warn': _meta_neg_streak >= warn_thresh,
        'meta_neg_streak_warn_threshold': warn_thresh,
        'meta_reg_slope': _meta_reg_slope,
        'meta_reg_pvalue': _meta_reg_pvalue,
        'meta_reg_window': int(getattr(settings, 'META_REG_WINDOW', 30)),
        'meta_coef_cosine': coef_cos,
        'meta_coef_drift_warn': drift_warn,
        'meta_coef_drift_warn_threshold': cos_warn,
    }


def can_trigger_prob_drift() -> bool:
    """Cooldown logic for probability drift triggered retrain."""
    if not getattr(settings, 'PROB_DRIFT_ENABLED', False):
        return False
    global _last_retrain_utc, _training_active
    if _training_active:
        return False
    if _last_retrain_utc is None:
        return True
    min_hours = float(getattr(settings, 'PROB_DRIFT_MIN_INTERVAL_HOURS', 12.0))
    diff = (datetime.now(tz=timezone.utc) - _last_retrain_utc).total_seconds() / 3600.0
    return diff >= min_hours


def trigger_prob_drift_retrain() -> bool:
    """Schedule retrain due to probability drift (async)."""
    if not can_trigger_prob_drift():
        return False
    base_days = getattr(settings, 'RETRAIN_DAYS', getattr(settings, 'BOTTOM_TRAIN_DAYS', 30))
    stacking_days = getattr(settings, 'RETRAIN_STACKING_DAYS', getattr(settings, 'STACKING_META_DAYS', base_days))
    async def _run():
        global _training_active
        if _training_active:
            return
        _training_active = True
        try:
            await run_training_sequence(base_days, stacking_days, reason='prob-drift')
        finally:
            _training_active = False
    loop = asyncio.get_event_loop(); loop.create_task(_run())
    logging.getLogger("scheduler").info("[event] prob-drift retrain scheduled base_days=%d stacking_days=%d", base_days, stacking_days)
    return True


# ---------------- Meta retrain (bottom-vs-forecast logistic layer) -----------------

def can_trigger_meta_retrain() -> bool:
    if not getattr(settings, 'META_RETRAIN_ENABLED', False):
        return False
    # Defer meta retrain if base training active (priority policy)
    global _training_active
    if _training_active:
        global _pending_meta_retrain
        _pending_meta_retrain = True
        return False
    global _last_meta_retrain_utc
    base_min = float(getattr(settings, 'META_RETRAIN_MIN_INTERVAL_MINUTES', 60.0))
    # Dynamic backoff: scale by (1 + streak * factor) if negative streak present
    factor = float(getattr(settings, 'META_BACKOFF_FACTOR', 0.5))
    min_interval_min = base_min * (1.0 + _meta_neg_streak * factor) if _meta_neg_streak > 0 and factor > 0 else base_min
    if _last_meta_retrain_utc is None:
        return True
    diff_min = (datetime.now(tz=timezone.utc) - _last_meta_retrain_utc).total_seconds() / 60.0
    return diff_min >= min_interval_min


async def run_meta_retrain_job() -> None:
    """APScheduler job: run logistic meta retrain script if cooldown satisfied.

    Invokes scripts.retrain_bottom_vs_forecast_meta with configured paths & threshold.
    Records whether overwrite occurred by parsing script stdout.
    """
    if not can_trigger_meta_retrain():
        log.info('[meta-retrain] cooldown active; skip')
        return
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    py = _python_executable(); env = os.environ.copy()
    csv_path = getattr(settings, 'BOTTOM_VS_FORECAST_EVAL_CSV_PATH', 'data/bottom_eval_sample.csv')
    meta_path = getattr(settings, 'BOTTOM_VS_FORECAST_META_PATH', 'data/bottom_vs_forecast_meta.json')
    min_rel = getattr(settings, 'META_RETRAIN_MIN_REL_BRIER_IMPROVE', 0.0)
    cmd = [py, '-m', 'scripts.retrain_bottom_vs_forecast_meta', '--csv', csv_path, '--current-meta', meta_path, '--out-meta', meta_path, '--min-rel-brier-improve', str(min_rel)]
    log.info('[meta-retrain] start csv=%s meta=%s min_rel=%.4f', csv_path, meta_path, float(min_rel))
    # Capture output for overwrite detection
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, cwd=repo_root, env=env)
    assert proc.stdout is not None
    overwritten = None
    async for line in proc.stdout:
        txt = line.decode(errors='ignore').rstrip()
        log.info('[meta-retrain] %s', txt)
        if '[meta] wrote new meta' in txt:
            overwritten = True
        if '[meta] NOT overwritten' in txt:
            overwritten = False
    rc = await proc.wait()
    log.info('[meta-retrain] exit code=%d overwritten=%s', rc, overwritten)
    # Reload registry to pick up updated meta (watcher should handle but force reload for immediacy)
    try:
        registry.load_from_settings()
        log.info('[meta-retrain] registry reloaded: %s', registry.status())
    except Exception as e:
        log.exception('[meta-retrain] registry reload failed: %s', e)
    global _last_meta_retrain_utc, _last_meta_retrain_reason, _last_meta_retrain_exit_code, _last_meta_retrain_overwritten
    _last_meta_retrain_utc = datetime.now(tz=timezone.utc)
    # Reason derived from scheduling mode
    if int(getattr(settings, 'META_RETRAIN_EVERY_MINUTES', 0) or 0) > 0:
        _last_meta_retrain_reason = 'interval'
    elif str(getattr(settings, 'META_RETRAIN_DAILY_AT', '') or '').strip():
        _last_meta_retrain_reason = 'daily'
    else:
        _last_meta_retrain_reason = 'manual-meta'
    _last_meta_retrain_exit_code = int(rc)
    _last_meta_retrain_overwritten = overwritten
    # Extract improvement metrics from output (already parsed via overwritten) – reload file to capture final metrics
    try:
        if os.path.exists(meta_path):
            import json as _json
            meta_obj = _json.loads(open(meta_path,'r',encoding='utf-8').read())
            brier_new = meta_obj.get('brier_new')
            prev = meta_obj.get('prev') or {}
            rel_imp = prev.get('rel_improve')
            brier_old = prev.get('brier_old')
            rec = {
                'ts': _last_meta_retrain_utc.isoformat() if _last_meta_retrain_utc else None,
                'brier_new': brier_new,
                'brier_old': brier_old,
                'rel_improve': rel_imp,
                'overwritten': overwritten,
                'samples': meta_obj.get('retrain_samples'),
            }
            _meta_history.append(rec)
            # Snapshot last good meta (positive improvement & overwritten True)
            try:
                if overwritten and isinstance(rel_imp, (int, float)) and rel_imp > 0:
                    snap_path = getattr(settings, 'META_LAST_GOOD_META_PATH', 'data/bottom_vs_forecast_meta_last_good.json')
                    import shutil
                    shutil.copyfile(meta_path, snap_path)
                    _last_good_meta_snapshot = snap_path
                    log.info('[meta-retrain] saved last good meta snapshot -> %s', snap_path)
            except Exception as _e:
                log.debug('[meta-retrain] snapshot copy failed: %s', _e)
            _compute_meta_history_stats()
            # Persist rolling history file (env override or default path)
            hist_path = getattr(settings, 'META_RETRAIN_HISTORY_PATH', 'data/bvf_meta_retrain_history.json')
            try:
                import json as _json2
                open(hist_path,'w',encoding='utf-8').write(_json2.dumps(_meta_history[-200:], ensure_ascii=False, indent=2))
            except Exception as _e:
                log.debug('[meta-retrain] history persist skipped: %s', _e)
    except Exception as _e:
        log.debug('[meta-retrain] history capture failed: %s', _e)
    # Optional rollback if excessive negative streak and snapshot available
    try:
        warn_thresh = int(getattr(settings, 'META_NEG_STREAK_WARN', 3))
        enable_rb = int(getattr(settings, 'META_NEG_STREAK_ROLLBACK', 0))
        if enable_rb and _meta_neg_streak >= warn_thresh and _last_good_meta_snapshot and os.path.exists(_last_good_meta_snapshot):
            import shutil
            shutil.copyfile(_last_good_meta_snapshot, meta_path)
            registry.load_from_settings()
            log.warning('[meta-retrain] rollback executed (neg streak=%d >= %d) -> restored %s', _meta_neg_streak, warn_thresh, _last_good_meta_snapshot)
    except Exception as _e:
        log.debug('[meta-retrain] rollback skipped: %s', _e)


# --------------- Evaluation CSV generation (daily rolling) ---------------
async def run_eval_csv_job() -> None:
    """Generate/refresh rolling evaluation CSV for bottom-vs-forecast meta.

    Strategy:
      - If target CSV exists, retain only last N days (default 30 via META_EVAL_KEEP_DAYS)
      - If missing, fallback to sample path or skip.
    """
    try:
        import pandas as _pd
        from datetime import datetime, timedelta
        target = getattr(settings, 'BOTTOM_VS_FORECAST_EVAL_CSV_PATH', 'data/bottom_eval_sample.csv')
        keep_days = int(getattr(settings, 'META_EVAL_KEEP_DAYS', 30))
        sample = 'data/bottom_eval_sample.csv'
        if not os.path.exists(target):
            if os.path.exists(sample):
                import shutil
                shutil.copyfile(sample, target)
                log.info('[eval-csv] initialized from sample -> %s', target)
                return
            else:
                log.warning('[eval-csv] target & sample missing; skip')
                return
        df = _pd.read_csv(target)
        if 'timestamp' in df.columns:
            try:
                df['ts'] = _pd.to_datetime(df['timestamp'])
                cutoff = datetime.utcnow() - timedelta(days=keep_days)
                before = len(df)
                df = df[df['ts'] >= cutoff].drop(columns=['ts'])
                df.to_csv(target, index=False)
                log.info('[eval-csv] pruned rows %d->%d keep_days=%d file=%s', before, len(df), keep_days, target)
            except Exception as _e:
                log.warning('[eval-csv] timestamp parse failed: %s', _e)
        else:
            log.info('[eval-csv] no timestamp column; untouched file=%s', target)
    except Exception as e:
        log.warning('[eval-csv] job failed: %s', e)

