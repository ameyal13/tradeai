$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\david\OneDrive\Desktop\TRADEAI"
$OpsWrapper = Join-Path $ProjectRoot "scripts\windows\run_shadow_ops_task.ps1"
$SummaryWrapper = Join-Path $ProjectRoot "scripts\windows\run_shadow_summary_task.ps1"
$TaskUser = "$env:USERDOMAIN\$env:USERNAME"

if (-not (Test-Path $OpsWrapper)) {
    throw "Missing wrapper: $OpsWrapper"
}
if (-not (Test-Path $SummaryWrapper)) {
    throw "Missing wrapper: $SummaryWrapper"
}

$OpsAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$OpsWrapper`"" `
    -WorkingDirectory $ProjectRoot

$OpsTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).Date.AddHours((Get-Date).Hour + 1) `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Days 3650)

$SummaryAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$SummaryWrapper`"" `
    -WorkingDirectory $ProjectRoot

$SummaryTrigger = New-ScheduledTaskTrigger -Daily -At 21:00

$Settings = New-ScheduledTaskSettingsSet `
    -RunOnlyIfNetworkAvailable `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName "TRADEAI Shadow Ops Hourly" `
    -Action $OpsAction `
    -Trigger $OpsTrigger `
    -Settings $Settings `
    -Description "TRADEAI research-only shadow ops cycle. No trading orders." `
    -User $TaskUser `
    -Force | Out-Null

Register-ScheduledTask `
    -TaskName "TRADEAI Shadow Summary Daily" `
    -Action $SummaryAction `
    -Trigger $SummaryTrigger `
    -Settings $Settings `
    -Description "TRADEAI research-only shadow summary notification. No trading orders." `
    -User $TaskUser `
    -Force | Out-Null

Write-Host "Installed Task Scheduler tasks:"
Write-Host "- TRADEAI Shadow Ops Hourly"
Write-Host "- TRADEAI Shadow Summary Daily"
Write-Host "Research only. No trading signal. No exchange orders."
Write-Host "Note: tasks are registered for the current Windows user without storing passwords."
