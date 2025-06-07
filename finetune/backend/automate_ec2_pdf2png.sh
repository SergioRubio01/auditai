#!/bin/bash

# Convert .env file to Unix format (fix \r issues)
sed -i 's/\r$//' .env

# Import environment variables
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found"
    exit 1
fi

# Validate required environment variables
required_vars=(
    "AMI_ID"
    "INSTANCE_TYPE"
    "KEY_FILE"
    "SECURITY_GROUP"
    "SUBNET_ID"
    "EC2_USER"
    "REMOTE_PDF_PATH"
    "LOCAL_PDF_PATH"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: Required environment variable $var is not set"
        exit 1
    fi
done

# 1. Launch EC2 instance
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "${KEY_FILE%.pem}" \
    --security-group-ids "$SECURITY_GROUP" \
    --subnet-id "$SUBNET_ID" \
    --query "Instances[0].InstanceId" \
    --output text)

if [ -z "$INSTANCE_ID" ]; then
    echo "Error: Failed to launch EC2 instance."
    exit 1
fi

echo "Launching EC2 instance: $INSTANCE_ID..."

# Wait for instance to be running
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
echo "EC2 instance is running."

# Get the correct public IP dynamically
EC2_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text)

if [ -z "$EC2_IP" ]; then
    echo "Error: Failed to retrieve EC2 public IP."
    exit 1
fi

echo "EC2 Public IP: $EC2_IP"

# Wait until instance status is OK
echo "Waiting for instance status checks to pass..."
aws ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID"
echo "Instance status checks passed."

# Wait for SSH to be available
echo "Waiting 30 seconds for instance to fully boot..."
sleep 30

# 2. Create necessary directories on EC2
echo "Creating directories on EC2..."
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_IP" "mkdir -p $REMOTE_PDF_PATH"

# 3. Upload pdf2png.py script
echo "Uploading pdf2png.py to EC2..."
scp -i "$KEY_FILE" "pdf2png.py" "$EC2_USER@$EC2_IP:/home/ubuntu/"

# 4. Upload PDF files
echo "Uploading PDF files to EC2..."
scp -i "$KEY_FILE" -r "$LOCAL_PDF_PATH"/* "$EC2_USER@$EC2_IP:$REMOTE_PDF_PATH/"

# 5. Install required Python packages
echo "Installing required Python packages..."
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_IP" <<EOF
    sudo apt-get update
    sudo apt-get install -y python3-pip poppler-utils
    pip3 install pdf2image PyPDF2 pandas pillow
EOF

# 6. Run the Python script on EC2
echo "Executing PDF processing script on EC2..."
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_IP" <<EOF
    cd /home/ubuntu
    python3 pdf2png.py -c "$REMOTE_PDF_PATH"
EOF

# 7. Retrieve processed files from EC2
echo "Retrieving processed files from EC2..."
scp -i "$KEY_FILE" -r "$EC2_USER@$EC2_IP:$REMOTE_PDF_PATH" "$LOCAL_PDF_PATH"

echo "PDF retrieval completed. Files are saved in: $LOCAL_PDF_PATH"

# 8. Terminate EC2 instance
echo "Terminating EC2 instance..."
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID"
echo "EC2 instance terminated."
