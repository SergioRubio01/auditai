# AutoAudit 2-Day Deployment Guide

## Overview
This accelerated guide deploys AutoAudit to AWS using ECS Fargate (serverless containers) in just 2 days. We'll use managed services to minimize setup time while maintaining production quality.

## Prerequisites Checklist

### Required Tools
- [ ] Docker 20.10+ installed
- [ ] Docker Compose 2.0+ installed
- [ ] AWS CLI configured
- [ ] Git configured
- [ ] Domain name for the application (optional)

### AWS Resources Needed
- [ ] AWS Account with appropriate permissions
- [ ] Default VPC (or create one)

## Day 1 Morning: Quick Setup (2 hours)

### 1.1 Environment Configuration
```bash
# Create environment files
cat > .env << 'EOF'
# Database
POSTGRES_USER=autoaudit
POSTGRES_PASSWORD=SecurePass123!
POSTGRES_DB=autoaudit

# Application
WANDB_API_KEY=your_wandb_key
LOG_LEVEL=INFO
WORKERS_COUNT=4
MAX_CONCURRENT_WORKFLOWS=10

# Redis
REDIS_PASSWORD=your_redis_password

# Monitoring
GRAFANA_PASSWORD=your_grafana_password
EOF
```

### 1.2 Database Migration
Since there are no existing Postgres models, create the initial schema:

```bash
# Create initial database schema file
mkdir -p infrastructure/postgres
cat > infrastructure/postgres/init.sql << 'EOF'
-- AutoAudit Database Schema
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Transferencias table
CREATE TABLE IF NOT EXISTS transferencias (
    id SERIAL PRIMARY KEY,
    concepto VARCHAR(500),
    fecha_valor DATE,
    importe DECIMAL(12,2),
    id_documento VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tarjetas table
CREATE TABLE IF NOT EXISTS tarjetas (
    id SERIAL PRIMARY KEY,
    numero_tarjeta VARCHAR(20),
    fecha_transaccion DATE,
    comercio VARCHAR(255),
    importe DECIMAL(12,2),
    id_documento VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Facturas table
CREATE TABLE IF NOT EXISTS facturas (
    id SERIAL PRIMARY KEY,
    cif_cliente VARCHAR(20),
    cliente VARCHAR(255),
    id_documento VARCHAR(255),
    numero_factura VARCHAR(50),
    fecha_factura DATE,
    proveedor VARCHAR(255),
    base_imponible DECIMAL(12,2),
    cif_proveedor VARCHAR(20),
    irpf DECIMAL(12,2),
    iva DECIMAL(12,2),
    total_factura DECIMAL(12,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Nominas table
CREATE TABLE IF NOT EXISTS nominas (
    id SERIAL PRIMARY KEY,
    id_documento VARCHAR(255),
    mes VARCHAR(20),
    fecha_inicio DATE,
    fecha_fin DATE,
    cif VARCHAR(20),
    trabajador VARCHAR(255),
    naf VARCHAR(50),
    nif VARCHAR(20),
    categoria VARCHAR(100),
    total_devengos DECIMAL(12,2),
    total_deducciones DECIMAL(12,2),
    liquido_a_percibir DECIMAL(12,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Workflow execution logs
CREATE TABLE IF NOT EXISTS workflow_executions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    workflow_id VARCHAR(255) NOT NULL,
    workflow_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    duration_seconds FLOAT,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_transferencias_documento ON transferencias(id_documento);
CREATE INDEX idx_tarjetas_documento ON tarjetas(id_documento);
CREATE INDEX idx_facturas_documento ON facturas(id_documento);
CREATE INDEX idx_nominas_documento ON nominas(id_documento);
CREATE INDEX idx_workflow_executions_workflow_id ON workflow_executions(workflow_id);
CREATE INDEX idx_workflow_executions_status ON workflow_executions(status);
EOF
```

### 1.3 Build and Test Locally
```bash
# Build the optimized Docker image
docker build -t autoaudit:latest .

# Quick local test
docker-compose up -d
curl http://localhost:8000/health

# If successful, stop local services
docker-compose down
```

## Day 1 Afternoon: AWS Infrastructure (4 hours)

### 2.1 Quick AWS Setup Script
Save this as `quick-setup.sh` and run it:

```bash
#!/bin/bash
# Quick AWS infrastructure setup

export AWS_REGION="us-east-1"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export APP_NAME="autoaudit"

echo "Setting up AutoAudit infrastructure..."

# 1. Create ECR Repository
aws ecr create-repository --repository-name $APP_NAME --region $AWS_REGION || true

# 2. Build and push Docker image
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
docker build -t ${APP_NAME}:latest .
docker tag ${APP_NAME}:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${APP_NAME}:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${APP_NAME}:latest

# 3. Create S3 Buckets
aws s3 mb s3://${APP_NAME}-uploads-${AWS_ACCOUNT_ID} || true
aws s3 mb s3://${APP_NAME}-backups-${AWS_ACCOUNT_ID} || true

# 4. Create RDS PostgreSQL (minimal config for speed)
aws rds create-db-instance \
    --db-instance-identifier ${APP_NAME}-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --master-username autoaudit \
    --master-user-password SecurePass123! \
    --allocated-storage 20 \
    --publicly-accessible || true

# 5. Create ECS Cluster for Fargate
aws ecs create-cluster --cluster-name ${APP_NAME}-cluster || true

# 6. Create Security Group
DEFAULT_VPC=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)
SG_ID=$(aws ec2 create-security-group \
    --group-name ${APP_NAME}-sg \
    --description "AutoAudit Security Group" \
    --vpc-id $DEFAULT_VPC \
    --query 'GroupId' \
    --output text)

aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 443 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 8000 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 8501 --cidr 0.0.0.0/0

echo "Infrastructure created! Security Group: $SG_ID"
```

### 2.2 Create ECS Task Definition
```bash
# Wait for RDS to be available (runs in background)
aws rds wait db-instance-available --db-instance-identifier ${APP_NAME}-db &

# Create CloudWatch Log Group
aws logs create-log-group --log-group-name /ecs/autoaudit || true

# Get RDS endpoint (after it's ready)
RDS_ENDPOINT=$(aws rds describe-db-instances \
    --db-instance-identifier ${APP_NAME}-db \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text)

# Create task definition
cat > task-definition.json << EOF
{
  "family": "autoaudit-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "autoaudit",
      "image": "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/autoaudit:latest",
      "portMappings": [
        {"containerPort": 8000, "protocol": "tcp"},
        {"containerPort": 8501, "protocol": "tcp"}
      ],
      "environment": [
        {"name": "DATABASE_URL", "value": "postgresql://autoaudit:SecurePass123!@${RDS_ENDPOINT}:5432/autoaudit"},
        {"name": "LOG_LEVEL", "value": "INFO"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/autoaudit",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
EOF

aws ecs register-task-definition --cli-input-json file://task-definition.json
```

## Day 2 Morning: Deployment (2 hours)

### 3.1 Quick Database Setup
```python
# Create PostgreSQL adapter for the application
cat > flow/backend/database/postgres_adapter.py << 'EOF'
import os
import asyncpg
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

class PostgresAdapter:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
    
    async def init_pool(self):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=10,
            command_timeout=60
        )
        logger.info("PostgreSQL connection pool initialized")
    
    async def close_pool(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool"""
        async with self.pool.acquire() as connection:
            yield connection

# Initialize on startup
database_url = os.getenv("DATABASE_URL", "postgresql://autoaudit:password@localhost:5432/autoaudit")
db = PostgresAdapter(database_url)
EOF
```

### 3.2 Deploy to ECS Fargate
```bash
# Get subnet information
SUBNETS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$DEFAULT_VPC" \
    --query "Subnets[*].SubnetId" \
    --output text | tr '\t' ',')

# Create ALB
ALB_ARN=$(aws elbv2 create-load-balancer \
    --name ${APP_NAME}-alb \
    --subnets $(echo $SUBNETS | tr ',' ' ') \
    --security-groups $SG_ID \
    --query 'LoadBalancers[0].LoadBalancerArn' \
    --output text)

# Create Target Groups
TG_API=$(aws elbv2 create-target-group \
    --name ${APP_NAME}-api \
    --protocol HTTP \
    --port 8000 \
    --vpc-id $DEFAULT_VPC \
    --target-type ip \
    --health-check-path /health \
    --query 'TargetGroups[0].TargetGroupArn' \
    --output text)

# Create ALB Listener
aws elbv2 create-listener \
    --load-balancer-arn $ALB_ARN \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=$TG_API

# Create ECS Service
aws ecs create-service \
    --cluster ${APP_NAME}-cluster \
    --service-name ${APP_NAME}-service \
    --task-definition autoaudit-task:1 \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
    --load-balancers targetGroupArn=$TG_API,containerName=autoaudit,containerPort=8000
```

## Day 2 Afternoon: Testing & Optimization (2 hours)

### 4.1 Run Database Migration
```bash
# Run migration once RDS is ready
PGPASSWORD=SecurePass123! psql -h $RDS_ENDPOINT -U autoaudit -d postgres -c "CREATE DATABASE autoaudit;"
PGPASSWORD=SecurePass123! psql -h $RDS_ENDPOINT -U autoaudit -d autoaudit -f infrastructure/postgres/init.sql
```

### 4.2 Test Deployment
```bash
# Get ALB DNS
ALB_DNS=$(aws elbv2 describe-load-balancers \
    --names ${APP_NAME}-alb \
    --query 'LoadBalancers[0].DNSName' \
    --output text)

# Test endpoints
echo "Testing endpoints..."
curl -f http://$ALB_DNS/health
echo "ALB URL: http://$ALB_DNS"

# Test load balancing
for i in {1..5}; do
    curl -s http://$ALB_DNS/health &
done
wait
echo "Load test completed"
```

### 4.3 Enable Auto-Scaling
```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --resource-id service/${APP_NAME}-cluster/${APP_NAME}-service \
    --scalable-dimension ecs:service:DesiredCount \
    --min-capacity 2 \
    --max-capacity 10

# Create scaling policy
cat > scaling-policy.json << EOF
{
  "TargetValue": 70.0,
  "PredefinedMetricSpecification": {
    "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
  }
}
EOF

aws application-autoscaling put-scaling-policy \
    --policy-name cpu-scaling \
    --service-namespace ecs \
    --resource-id service/${APP_NAME}-cluster/${APP_NAME}-service \
    --scalable-dimension ecs:service:DesiredCount \
    --policy-type TargetTrackingScaling \
    --target-tracking-scaling-policy-configuration file://scaling-policy.json
```

### 4.4 Quick Monitoring Setup
```bash
# Create CloudWatch Dashboard
cat > dashboard.json << 'EOF'
{
    "widgets": [
        {
            "type": "metric",
            "properties": {
                "metrics": [
                    ["AWS/ECS", "CPUUtilization", "ServiceName", "autoaudit-service"],
                    [".", "MemoryUtilization", ".", "."]
                ],
                "period": 300,
                "stat": "Average",
                "region": "us-east-1",
                "title": "Service Metrics"
            }
        }
    ]
}
EOF

aws cloudwatch put-dashboard --dashboard-name AutoAudit --dashboard-body file://dashboard.json

# Create basic alarm
aws cloudwatch put-metric-alarm \
    --alarm-name autoaudit-high-cpu \
    --alarm-description "Alarm when CPU exceeds 80%" \
    --metric-name CPUUtilization \
    --namespace AWS/ECS \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2
```

## Quick Reference

### Service Management
```bash
# Update service
aws ecs update-service --cluster ${APP_NAME}-cluster --service ${APP_NAME}-service --force-new-deployment

# Scale service
aws ecs update-service --cluster ${APP_NAME}-cluster --service ${APP_NAME}-service --desired-count 5

# View logs
aws logs tail /ecs/autoaudit --follow
```

### Troubleshooting
```bash
# Check ECS service
aws ecs describe-services --cluster ${APP_NAME}-cluster --services ${APP_NAME}-service

# View task details
TASK_ARN=$(aws ecs list-tasks --cluster ${APP_NAME}-cluster --service ${APP_NAME}-service --query 'taskArns[0]' --output text)
aws ecs describe-tasks --cluster ${APP_NAME}-cluster --tasks $TASK_ARN

# Quick database backup
aws rds create-db-snapshot --db-instance-identifier ${APP_NAME}-db --db-snapshot-identifier ${APP_NAME}-snapshot-$(date +%Y%m%d)
```

## Security Checklist

- [ ] Change default RDS password
- [ ] Restrict security group access
- [ ] Enable RDS encryption
- [ ] Configure SSL for ALB
- [ ] Set up IAM roles properly
- [ ] Enable VPC Flow Logs
- [ ] Configure S3 bucket policies

## Cost Summary

### Estimated Monthly Costs (AWS ECS Fargate)
- ECS Fargate (2 tasks @ 0.5 vCPU, 1GB): ~$30
- RDS PostgreSQL (t3.micro, single-AZ): ~$15
- Application Load Balancer: ~$25
- S3 storage: ~$10
- Data transfer: ~$15
- **Total: ~$95/month**

### Cost Saving Tips
1. Use Savings Plans for 20% savings
2. Right-size tasks based on metrics
3. Use Spot capacity for non-production
4. Enable S3 lifecycle policies

## Post-Deployment Tasks

### Immediate (Day 3)
1. Configure custom domain in Route 53
2. Add SSL certificate via ACM
3. Test all workflows
4. Set up backup automation

### Within 1 Week
1. Set up comprehensive monitoring
2. Implement CI/CD pipeline
3. Configure log aggregation
4. Performance testing

### Within 1 Month
1. Optimize costs with Savings Plans
2. Implement disaster recovery
3. Add advanced monitoring (Prometheus/Grafana)
4. Security audit and hardening

## Emergency Procedures

### Quick Rollback
```bash
# Rollback to previous task definition
aws ecs update-service --cluster ${APP_NAME}-cluster --service ${APP_NAME}-service --task-definition autoaudit-task:PREVIOUS_VERSION
```

### Database Recovery
```bash
# Restore from snapshot
aws rds restore-db-instance-from-db-snapshot --db-instance-identifier ${APP_NAME}-db-restored --db-snapshot-identifier SNAPSHOT_ID
```

## Complete Setup Script

Run the `quick-setup.sh` script on Day 1, then execute the deployment commands on Day 2. The total setup time is approximately:

- **Day 1**: 6 hours (setup + infrastructure)
- **Day 2**: 4 hours (deployment + testing)

Your application will be live and accessible via the ALB URL by the end of Day 2!