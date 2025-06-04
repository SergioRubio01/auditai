$account = aws sts get-caller-identity --query Account --output text
$region = "eu-west-3"
Write-Host "Bootstrapping CDK for account: $account in region: $region"
npx cdk bootstrap "aws://$account/$region" 