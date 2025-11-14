Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $ScriptDir '..')

Write-Host 'Building images (shared layers)...'
docker compose build

Write-Host 'Starting stack with dev override (backend --reload, DEBUG, loops off)...'
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

function Test-Health {
  param(
    [string]$Url,
    [int]$Retries = 40,
    [int]$DelaySeconds = 2
  )
  for ($i = 0; $i -lt $Retries; $i++) {
    try {
      $r = Invoke-RestMethod -Uri $Url -TimeoutSec 5
      if ($r.status -eq 'ok') { return $true }
    } catch { }
    Start-Sleep -Seconds $DelaySeconds
  }
  return $false
}

Write-Host 'Waiting for backend health on http://localhost:8000/health...'
if (-not (Test-Health -Url 'http://localhost:8000/health')) {
  Write-Warning 'Backend health check did not become OK in time.'
} else {
  Write-Host 'Backend is healthy.'
}

Write-Host 'Frontend status code check on http://localhost:8080 ...'
try {
  $code = (Invoke-WebRequest 'http://localhost:8080' -UseBasicParsing -TimeoutSec 5).StatusCode
  Write-Host "Frontend HTTP status: $code"
} catch {
  Write-Warning 'Frontend is not reachable yet.'
}
