$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\david\OneDrive\Desktop\TRADEAI"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$OpsWrapper = Join-Path $ProjectRoot "scripts\windows\run_shadow_ops_task.ps1"
$LogDir = Join-Path $ProjectRoot "logs\shadow_ops"

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "Running TRADEAI Shadow Ops manual task test"
Write-Host "Research only. No trading signal. No exchange orders."
Write-Host ""

Write-Host "Step 1: healthcheck + Telegram test"
& $Python "scripts\shadow_ops_healthcheck.py" --check-news-context --test-telegram
$HealthExit = $LASTEXITCODE
if ($HealthExit -ne 0) {
    throw "Healthcheck failed with exit code $HealthExit"
}

Write-Host ""
Write-Host "Step 2: run shadow ops wrapper once"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $OpsWrapper
$OpsExit = $LASTEXITCODE
if ($OpsExit -ne 0) {
    throw "Shadow ops wrapper failed with exit code $OpsExit"
}

Write-Host ""
Write-Host "Logs directory:"
Write-Host $LogDir
Write-Host ""
Write-Host "Recent log files:"
Get-ChildItem $LogDir -Filter "*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 5 FullName, LastWriteTime
Write-Host ""
Write-Host "Manual test completed. No scheduler task was registered."
