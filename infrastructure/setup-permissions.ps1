# Attach required AWS managed policies to the textract-user
Write-Host "Attaching required policies to textract-user..."

# Try to attach CloudFormation and IAM policies first
$essentialPolicies = @(
    "arn:aws:iam::aws:policy/AWSCloudFormationFullAccess",
    "arn:aws:iam::aws:policy/IAMFullAccess"
)

foreach ($policy in $essentialPolicies) {
    Write-Host "Attaching policy: $policy"
    aws iam attach-user-policy --user-name textract-user --policy-arn $policy
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to attach policy: $policy"
        Write-Host "Please ask your AWS administrator to grant you these permissions."
        exit 1
    }
}

Write-Host "Basic permissions attached. Please wait a few minutes for the permissions to propagate."
Write-Host "Then run .\deploy.ps1 again." 