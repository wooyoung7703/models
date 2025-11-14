$logDir = "backend\\logs"
if (!(Test-Path -LiteralPath $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$logPath = Join-Path $logDir "server_$ts.log"
$env:EXCHANGE_TYPE = 'futures'
$env:ENABLE_ADAPTIVE_THRESHOLD = '1'
$env:ADAPTIVE_THRESHOLD_QUANTILE = '0.88'
$env:ADD_COOLDOWN_SECONDS = '120'
$env:TP_MODE = 'trailing'
$env:TP_TRIGGER = '0.005'
$env:TP_STEP = '0.0005'
$env:TP_GIVEBACK = '0.001'
$env:STACKING_THRESHOLD = '-1'
$env:PREDICT_INTERVAL_SECONDS = '10'
$env:LOG_LEVEL = 'INFO'
$env:LOG_FORMAT = 'json'
$cmd = "python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 >> `"$logPath`" 2>>&1"
Start-Process -FilePath cmd.exe -ArgumentList '/c', $cmd -WindowStyle Hidden
Start-Sleep -Seconds 3
if (Test-Path -LiteralPath $logPath) {
    Get-Content -Path $logPath -Tail 200
} else {
    Write-Host "Log file not found yet: $logPath"
}
