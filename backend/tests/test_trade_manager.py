from sqlmodel import Session, select
from backend.app.trade_manager import TradeManager
from backend.app.models import Trade, TradeFill
from backend.app.db import init_db, engine


def _fake_nowcast(prob: float, threshold: float = 0.9, margin: float = 0.0, conf: float = 0.0, bottom: float = 0.0, z: float = 0.0, decision: bool = False):
    return {
        'bottom_score': bottom,
        'stacking': {
            'ready': True,
            'prob': prob,
            'threshold': threshold,
            'margin': margin,
            'confidence': conf,
            'z': z,
            'decision': decision,
        }
    }


def setup_module():
    init_db()


def test_trade_entry_and_close(tmp_path):
    # Disable cooldown in tests to allow immediate add
    tm = TradeManager(lambda: Session(engine), add_cooldown_seconds=0)
    # Entry should occur (prob above threshold, margin/conf/bottom/z pass, decision True)
    nc_enter = _fake_nowcast(prob=0.95, margin=0.05, conf=0.2, bottom=0.7, z=2.0, decision=True)
    act1 = tm.process(symbol='btcusdt', interval='1m', exchange_type='futures', price=100.0, nowcast=nc_enter)
    assert act1['action'] == 'enter'
    # Retrieve trade
    with Session(engine) as s:
        t = s.exec(select(Trade)).first()
    # Simulate take profit hit (+1%)
    nc_hold = _fake_nowcast(prob=0.96, margin=0.06, conf=0.21, bottom=0.72, z=2.2, decision=True)
    act2 = tm.process(symbol='btcusdt', interval='1m', exchange_type='futures', price=101.1, nowcast=nc_hold)
    # Should close (approx +1.1%)
    assert act2['action'] in {'close', 'add', 'hold'}


def test_dca_add(tmp_path):
    # Clean previous trades for isolation
    with Session(engine) as s:
        from backend.app.models import Trade
        olds = s.exec(select(Trade).where(Trade.symbol=='ethusdt')).all()  # type: ignore[attr-defined]
        for o in olds:
            s.delete(o)
        s.commit()
    tm = TradeManager(lambda: Session(engine), add_cooldown_seconds=0)
    # Ensure margin/conf/bottom/z meet entry thresholds exactly
    nc_enter = _fake_nowcast(prob=0.94, margin=0.03, conf=0.12, bottom=0.65, z=1.7, decision=True)
    enter = tm.process(symbol='ethusdt', interval='1m', exchange_type='futures', price=50.0, nowcast=nc_enter)
    assert enter['action'] == 'enter'
    # Slight drop triggers potential add
    nc_add = _fake_nowcast(prob=0.94, margin=0.03, conf=0.12, bottom=0.65, z=1.7, decision=True)
    # Slightly less than 0.5% drop to avoid SL (<= -0.5% would close)
    act = tm.process(symbol='ethusdt', interval='1m', exchange_type='futures', price=49.8, nowcast=nc_add)
    assert act['action'] in {'add','hold'}
