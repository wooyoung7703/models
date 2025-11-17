# Compromise live mode startup script
# Futures mode, entry-meta gate ON, no stop loss, fixed TP +0.8% (net-of-fees)
# Uses WebSocket server (no FastAPI HTTP) on default port 8022.

$env:EXCHANGE_TYPE='futures'
$env:ENABLE_ENTRY_META='1'
$env:ENTRY_META_GATE_ENABLED='1'
$env:DISABLE_STOP_LOSS='1'
$env:TAKE_PROFIT_PCT='0.008'
$env:TP_MODE='fixed'
$env:TP_DECISION_ON_NET='1'
# Optional guard rails (uncomment as needed)
# $env:ADD_COOLDOWN_SECONDS='900'
# Light mode off to allow full model stack (set to '1' for lightweight run)
$env:LIGHT_MODE='0'

python -m backend.app.ws_server
