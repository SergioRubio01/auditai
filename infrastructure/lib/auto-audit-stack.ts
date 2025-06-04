import * as cdk from 'aws-cdk-lib';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as elasticache from 'aws-cdk-lib/aws-elasticache';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as certificatemanager from 'aws-cdk-lib/aws-certificatemanager';
import * as route53 from 'aws-cdk-lib/aws-route53';
import * as targets from 'aws-cdk-lib/aws-route53-targets';
import { Construct } from 'constructs';

export class AutoAuditStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const hostedZone = route53.HostedZone.fromLookup(this, 'HostedZone', {
      domainName: 'bizai.es',
    });

    const certificate = certificatemanager.Certificate.fromCertificateArn(
      this,
      'BizaiCert',
      'arn:aws:acm:eu-west-3:533267139503:certificate/c185ff2f-2f8b-4cc8-8137-df2067db38f1'
    );

    const vpc = new ec2.Vpc(this, 'AutoAuditVPC', {
      maxAzs: 2,
      natGateways: 1,
    });

    const repository = ecr.Repository.fromRepositoryName(this, 'AutoAuditRepo', 'auto-audit');

    const cluster = new ecs.Cluster(this, 'AutoAuditCluster', {
      vpc,
      clusterName: 'auto-audit-cluster',
    });

    const redisSubnetGroup = new elasticache.CfnSubnetGroup(this, 'RedisSubnetGroup', {
      subnetIds: vpc.privateSubnets.map(subnet => subnet.subnetId),
      description: 'Subnet group for Redis cache',
    });

    const redisSecurityGroup = new ec2.SecurityGroup(this, 'RedisSecurityGroup', {
      vpc,
      description: 'Security group for Redis cache',
      allowAllOutbound: true,
    });

    const redis = new elasticache.CfnCacheCluster(this, 'RedisCluster', {
      engine: 'redis',
      cacheNodeType: 'cache.t3.micro',
      numCacheNodes: 1,
      vpcSecurityGroupIds: [redisSecurityGroup.securityGroupId],
      cacheSubnetGroupName: redisSubnetGroup.ref,
    });

    const taskRole = new iam.Role(this, 'AutoAuditTaskRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
    });

    taskRole.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonECSTaskExecutionRolePolicy')
    );

    const taskDefinition = new ecs.FargateTaskDefinition(this, 'AutoAuditTaskDef', {
      memoryLimitMiB: 2048,
      cpu: 1024,
      taskRole,
      family: 'auto-audit',
    });

    const container = taskDefinition.addContainer('AutoAuditContainer', {
      image: ecs.ContainerImage.fromEcrRepository(repository),
      memoryLimitMiB: 2048,
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: 'auto-audit',
        logRetention: logs.RetentionDays.ONE_MONTH,
      }),
      environment: {
        REDIS_HOST: redis.attrRedisEndpointAddress,
        REDIS_PORT: redis.attrRedisEndpointPort,
      },
    });

    container.addPortMappings(
      { containerPort: 8000, protocol: ecs.Protocol.TCP },
      { containerPort: 8501, protocol: ecs.Protocol.TCP }
    );

    const serviceSg = new ec2.SecurityGroup(this, 'ServiceSecurityGroup', {
      vpc,
      description: 'Security group for Auto Audit Service',
      allowAllOutbound: true,
    });

    // Allow inbound traffic from ALB to ECS service
    serviceSg.addIngressRule(
      serviceSg,
      ec2.Port.tcp(8000),
      'Allow inbound traffic to API'
    );

    serviceSg.addIngressRule(
      serviceSg,
      ec2.Port.tcp(8501),
      'Allow inbound traffic to Streamlit'
    );

    const service = new ecs.FargateService(this, 'AutoAuditService', {
      cluster,
      taskDefinition,
      desiredCount: 1,
      assignPublicIp: false,
      securityGroups: [serviceSg],
      vpcSubnets: {
        subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS
      }
    });

    const lb = new elbv2.ApplicationLoadBalancer(this, 'AutoAuditALB', {
      vpc,
      internetFacing: true,
      securityGroup: serviceSg,
    });

    // Create single HTTPS listener
    const httpsListener = lb.addListener('HttpsListener', {
      port: 443,
      protocol: elbv2.ApplicationProtocol.HTTPS,
      certificates: [certificate],
      defaultAction: elbv2.ListenerAction.fixedResponse(404),
    });

    // Add API target group
    const apiTargetGroup = httpsListener.addTargets('ApiTarget', {
      port: 8000,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targets: [
        service.loadBalancerTarget({
          containerName: 'AutoAuditContainer',
          containerPort: 8000,
        }),
      ],
      conditions: [
        elbv2.ListenerCondition.hostHeaders(['api.bizai.es']),
      ],
      priority: 1,
      healthCheck: {
        path: '/health',
        unhealthyThresholdCount: 2,
        healthyThresholdCount: 5,
        interval: cdk.Duration.seconds(30),
      },
    });

    // Add Streamlit target group
    const streamlitTargetGroup = httpsListener.addTargets('StreamlitTarget', {
      port: 8501,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targets: [
        service.loadBalancerTarget({
          containerName: 'AutoAuditContainer',
          containerPort: 8501,
        }),
      ],
      conditions: [
        elbv2.ListenerCondition.hostHeaders(['bizai.es']),
      ],
      priority: 2,
      healthCheck: {
        path: '/',
        unhealthyThresholdCount: 2,
        healthyThresholdCount: 5,
        interval: cdk.Duration.seconds(30),
      },
    });

    // Create DNS records for both services
    new route53.ARecord(this, 'ApiAliasRecord', {
      zone: hostedZone,
      recordName: 'api',
      target: route53.RecordTarget.fromAlias(new targets.LoadBalancerTarget(lb)),
    });

    new route53.ARecord(this, 'AppAliasRecord', {
      zone: hostedZone,
      recordName: '',
      target: route53.RecordTarget.fromAlias(new targets.LoadBalancerTarget(lb)),
    });

    // Output the URLs for easy access
    new cdk.CfnOutput(this, 'StreamlitURL', {
      value: 'https://bizai.es',
      description: 'URL for the Streamlit application',
    });

    new cdk.CfnOutput(this, 'ApiURL', {
      value: 'https://api.bizai.es',
      description: 'URL for the API',
    });

    new cdk.CfnOutput(this, 'LoadBalancerDNS', {
      value: lb.loadBalancerDnsName,
    });

    new cdk.CfnOutput(this, 'RepositoryURI', {
      value: repository.repositoryUri,
    });
  }
}
