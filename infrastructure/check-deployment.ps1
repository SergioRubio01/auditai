# Check CloudFormation stack status
Write-Host "Checking CloudFormation stack status..."
$stackStatus = aws cloudformation describe-stacks --stack-name AutoAuditStack --query 'Stacks[0].StackStatus' --output text
Write-Host "Stack Status: $stackStatus"

if ($stackStatus -eq "CREATE_COMPLETE") {
    # Get the Load Balancer DNS
    Write-Host "`nGetting Load Balancer DNS..."
    $lbDns = aws cloudformation describe-stacks --stack-name AutoAuditStack --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDNS`].OutputValue' --output text
    
    Write-Host "`nYour application endpoints:"
    Write-Host "API (FastAPI): http://$lbDns/"
    Write-Host "Dashboard (Streamlit): http://$lbDns:8501/"
    
    # Check ECS service status
    Write-Host "`nChecking ECS service status..."
    $cluster = "auto-audit-cluster"
    $services = aws ecs list-services --cluster $cluster --query 'serviceArns[]' --output text
    
    if ($services) {
        foreach ($service in $services.Split()) {
            $serviceStatus = aws ecs describe-services --cluster $cluster --services $service --query 'services[0].status' --output text
            Write-Host "ECS Service Status: $serviceStatus"
        }
    } else {
        Write-Host "No ECS services found in cluster $cluster"
    }
} else {
    Write-Host "Stack is not ready yet. Current status: $stackStatus"
    Write-Host "Please wait for the stack to complete deployment."
} 