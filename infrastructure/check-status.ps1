# Get stack status
$stack = aws cloudformation describe-stacks --stack-name AutoAuditStack --output json | ConvertFrom-Json
Write-Host "Stack Status: $($stack.Stacks[0].StackStatus)"

# Get recent events
Write-Host "`nRecent Events:"
$events = aws cloudformation describe-stack-events --stack-name AutoAuditStack --output json | ConvertFrom-Json
$events.StackEvents | Select-Object -First 5 | ForEach-Object {
    Write-Host "Resource: $($_.LogicalResourceId)"
    Write-Host "Status: $($_.ResourceStatus)"
    if ($_.ResourceStatusReason) {
        Write-Host "Reason: $($_.ResourceStatusReason)"
    }
    Write-Host "---"
} 