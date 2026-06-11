$ErrorActionPreference = "Stop"

$TaskNames = @(
    "TRADEAI Shadow Ops Hourly",
    "TRADEAI Shadow Summary Daily"
)

foreach ($TaskName in $TaskNames) {
    $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -ne $Task) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed task: $TaskName"
    }
    else {
        Write-Host "Task not found: $TaskName"
    }
}

Write-Host "Shadow ops Task Scheduler cleanup complete."
