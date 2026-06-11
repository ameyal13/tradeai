$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\david\OneDrive\Desktop\TRADEAI"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogDir = Join-Path $ProjectRoot "logs\shadow_ops"
$DateStamp = Get-Date -Format "yyyyMMdd"
$LogPath = Join-Path $LogDir "shadow_ops_$DateStamp.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $ProjectRoot

Add-Content -Path $LogPath -Value ""
Add-Content -Path $LogPath -Value "[$(Get-Date -Format o)] TRADEAI Shadow Ops task started"
Add-Content -Path $LogPath -Value "Research only. No trading signal. No exchange order."

try {
    $Output = & $Python "scripts\run_shadow_ops_once.py" `
        --max-signals 1 `
        --max-configs-scanned 21 `
        --use-news-context `
        --notify-telegram 2>&1
    $ExitCode = $LASTEXITCODE
    $Output | Out-File -FilePath $LogPath -Append -Encoding utf8
    Add-Content -Path $LogPath -Value "[$(Get-Date -Format o)] TRADEAI Shadow Ops task finished exit_code=$ExitCode"
    exit $ExitCode
}
catch {
    Add-Content -Path $LogPath -Value "[$(Get-Date -Format o)] TRADEAI Shadow Ops task failed: $($_.Exception.GetType().Name): $($_.Exception.Message)"
    exit 1
}
