# Get the ECR password
$ecrPassword = aws ecr get-login-password --region eu-west-3
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to get ECR password"
    exit 1
}

# Login to ECR
$ecrPassword | docker login --username AWS --password-stdin 533267139503.dkr.ecr.eu-west-3.amazonaws.com

# Build the image
Write-Host "Building Docker image..."
docker build -t auto-audit ..

# Tag the image
Write-Host "Tagging image..."
docker tag auto-audit:latest 533267139503.dkr.ecr.eu-west-3.amazonaws.com/auto-audit:latest

# Push to ECR
Write-Host "Pushing image to ECR..."
docker push 533267139503.dkr.ecr.eu-west-3.amazonaws.com/auto-audit:latest 