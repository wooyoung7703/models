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
    _scheduler.start()
    log.info("[scheduler] started; next run(s): %s", _scheduler.get_jobs())
    return _scheduler


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
    return {
        'last_retrain_utc': _last_retrain_utc.isoformat() if _last_retrain_utc else None,
        'last_retrain_reason': _last_retrain_reason,
        'last_retrain_base_days': _last_retrain_base_days,
        'last_retrain_stacking_days': _last_retrain_stacking_days,
        'training_active': _training_active,
        'last_retrain_type': _last_retrain_type,
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

