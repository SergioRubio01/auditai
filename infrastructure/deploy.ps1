# Clean up build artifacts and CDK outputs
Write-Host "Cleaning up build artifacts and CDK outputs..."
Remove-Item -Path "cdk.out" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "dist" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "cdk.context.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "cdk-outputs.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "tsconfig.tsbuildinfo" -Force -ErrorAction SilentlyContinue

# Note: Preserving package.json, package-lock.json, and tsconfig.json as they are configuration files

# Install dependencies
Write-Host "Installing dependencies..."
npm install

# Build the TypeScript code
Write-Host "Building TypeScript code..."
npm run build

# Get AWS account ID
$account = aws sts get-caller-identity --query Account --output text
$region = "eu-west-3"

Write-Host "Bootstrapping CDK for account: $account in region: $region"
npx cdk bootstrap "aws://$account/$region"

# Destroy existing stack
Write-Host "Destroying existing stack..."
npx cdk destroy --force

# Clean up any orphaned resources
Write-Host "Cleaning up any orphaned resources..."
$cluster = "auto-audit-cluster"
$services = aws ecs list-services --cluster $cluster --query 'serviceArns[]' --output text

if ($services) {
    Write-Host "Found existing services, cleaning up..."
    foreach ($service in $services.Split()) {
        $serviceName = $service.Split('/')[-1]
        Write-Host "Updating service $serviceName to 0 tasks..."
        aws ecs update-service --cluster $cluster --service $serviceName --desired-count 0
        Write-Host "Waiting for tasks to drain..."
        aws ecs wait services-stable --cluster $cluster --services $serviceName
        Write-Host "Deleting service $serviceName..."
        aws ecs delete-service --cluster $cluster --service $serviceName --force
    }
}

Write-Host "Deploying new stack..."
npx cdk deploy --require-approval never --outputs-file ./cdk-outputs.json

# Get the service name from the cluster
Write-Host "Getting service name..."
$serviceName = aws ecs list-services --cluster auto-audit-cluster --query 'serviceArns[0]' --output text | Select-String -Pattern '[^/]+$' | ForEach-Object { $_.Matches[0].Value }
Write-Host "Found service name: $serviceName"

# Get the latest task definition ARN
Write-Host "Getting latest task definition..."
$taskDefinition = aws ecs list-task-definitions --family-prefix auto-audit --sort DESC --max-items 1 --query 'taskDefinitionArns[0]' --output text

Write-Host "Updating ECS service with latest task definition: $taskDefinition"
aws ecs update-service `
    --cluster auto-audit-cluster `
    --service $serviceName `
    --task-definition $taskDefinition `
    --force-new-deployment

Write-Host "Deployment complete. Waiting for service to stabilize..."
aws ecs wait services-stable --cluster auto-audit-cluster --services $serviceName
Write-Host "Service is stable"
