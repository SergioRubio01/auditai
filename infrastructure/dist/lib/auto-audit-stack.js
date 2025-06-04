"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.AutoAuditStack = void 0;
const cdk = require("aws-cdk-lib");
const ecr = require("aws-cdk-lib/aws-ecr");
const ecs = require("aws-cdk-lib/aws-ecs");
const ec2 = require("aws-cdk-lib/aws-ec2");
const elbv2 = require("aws-cdk-lib/aws-elasticloadbalancingv2");
const iam = require("aws-cdk-lib/aws-iam");
const elasticache = require("aws-cdk-lib/aws-elasticache");
const logs = require("aws-cdk-lib/aws-logs");
const certificatemanager = require("aws-cdk-lib/aws-certificatemanager");
const route53 = require("aws-cdk-lib/aws-route53");
const targets = require("aws-cdk-lib/aws-route53-targets");
class AutoAuditStack extends cdk.Stack {
    constructor(scope, id, props) {
        super(scope, id, props);
        const hostedZone = route53.HostedZone.fromLookup(this, 'HostedZone', {
            domainName: 'bizai.es',
        });
        const certificate = certificatemanager.Certificate.fromCertificateArn(this, 'BizaiCert', 'arn:aws:acm:eu-west-3:533267139503:certificate/c185ff2f-2f8b-4cc8-8137-df2067db38f1');
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
        taskRole.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonECSTaskExecutionRolePolicy'));
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
        container.addPortMappings({ containerPort: 8000, protocol: ecs.Protocol.TCP }, { containerPort: 8501, protocol: ecs.Protocol.TCP });
        const serviceSg = new ec2.SecurityGroup(this, 'ServiceSecurityGroup', {
            vpc,
            description: 'Security group for Auto Audit Service',
            allowAllOutbound: true,
        });
        // Allow inbound traffic from ALB to ECS service
        serviceSg.addIngressRule(serviceSg, ec2.Port.tcp(8000), 'Allow inbound traffic to API');
        serviceSg.addIngressRule(serviceSg, ec2.Port.tcp(8501), 'Allow inbound traffic to Streamlit');
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
            recordName: 'app',
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
exports.AutoAuditStack = AutoAuditStack;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYXV0by1hdWRpdC1zdGFjay5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uL2xpYi9hdXRvLWF1ZGl0LXN0YWNrLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7OztBQUFBLG1DQUFtQztBQUNuQywyQ0FBMkM7QUFDM0MsMkNBQTJDO0FBQzNDLDJDQUEyQztBQUMzQyxnRUFBZ0U7QUFDaEUsMkNBQTJDO0FBQzNDLDJEQUEyRDtBQUMzRCw2Q0FBNkM7QUFDN0MseUVBQXlFO0FBQ3pFLG1EQUFtRDtBQUNuRCwyREFBMkQ7QUFHM0QsTUFBYSxjQUFlLFNBQVEsR0FBRyxDQUFDLEtBQUs7SUFDM0MsWUFBWSxLQUFnQixFQUFFLEVBQVUsRUFBRSxLQUFzQjtRQUM5RCxLQUFLLENBQUMsS0FBSyxFQUFFLEVBQUUsRUFBRSxLQUFLLENBQUMsQ0FBQztRQUV4QixNQUFNLFVBQVUsR0FBRyxPQUFPLENBQUMsVUFBVSxDQUFDLFVBQVUsQ0FBQyxJQUFJLEVBQUUsWUFBWSxFQUFFO1lBQ25FLFVBQVUsRUFBRSxVQUFVO1NBQ3ZCLENBQUMsQ0FBQztRQUVILE1BQU0sV0FBVyxHQUFHLGtCQUFrQixDQUFDLFdBQVcsQ0FBQyxrQkFBa0IsQ0FDbkUsSUFBSSxFQUNKLFdBQVcsRUFDWCxxRkFBcUYsQ0FDdEYsQ0FBQztRQUVGLE1BQU0sR0FBRyxHQUFHLElBQUksR0FBRyxDQUFDLEdBQUcsQ0FBQyxJQUFJLEVBQUUsY0FBYyxFQUFFO1lBQzVDLE1BQU0sRUFBRSxDQUFDO1lBQ1QsV0FBVyxFQUFFLENBQUM7U0FDZixDQUFDLENBQUM7UUFFSCxNQUFNLFVBQVUsR0FBRyxHQUFHLENBQUMsVUFBVSxDQUFDLGtCQUFrQixDQUFDLElBQUksRUFBRSxlQUFlLEVBQUUsWUFBWSxDQUFDLENBQUM7UUFFMUYsTUFBTSxPQUFPLEdBQUcsSUFBSSxHQUFHLENBQUMsT0FBTyxDQUFDLElBQUksRUFBRSxrQkFBa0IsRUFBRTtZQUN4RCxHQUFHO1lBQ0gsV0FBVyxFQUFFLG9C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a0JBQWtCLENBQUMsRUFBRSxDQUFDLENBQUM7U0FDM0UsQ0FBQyxDQUFDO1FBRUgsSUFBSSxPQUFPLENBQUMsT0FBTyxDQUFDLElBQUksRUFBRSxnQkFBZ0IsRUFBRTtZQUMxQyxJQUFJLEVBQUUsVUFBVTtZQUNoQixVQUFVLEVBQUUsS0FBSztZQUNqQixNQUFNLEVBQUUsT0FBTyxDQUFDLFlBQVksQ0FBQyxTQUFTLENBQUMsSUFBSSxPQUFPLENBQUMsa0JBQWtCLENBQUMsRUFBRSxDQUFDLENBQUM7U0FDM0UsQ0FBQyxDQUFDO1FBRUgsa0NBQWtDO1FBQ2xDLElBQUksR0FBRyxDQUFDLFNBQVMsQ0FBQyxJQUFJLEVBQUUsY0FBYyxFQUFFO1lBQ3RDLEtBQUssRUFBRSxrQkFBa0I7WUFDekIsV0FBVyxFQUFFLG1DQUFtQztTQUNqRCxDQUFDLENBQUM7UUFFSCxJQUFJLEdBQUcsQ0FBQyxTQUFTLENBQUMsSUFBSSxFQUFFLFFBQVEsRUFBRTtZQUNoQyxLQUFLLEVBQUUsc0JBQXNCO1lBQzdCLFdBQVcsRUFBRSxpQkFBaUI7U0FDL0IsQ0FBQyxDQUFDO1FBRUgsSUFBSSxHQUFHLENBQUMsU0FBUyxDQUFDLElBQUksRUFBRSxpQkFBaUIsRUFBRTtZQUN6QyxLQUFLLEVBQUUsRUFBRSxDQUFDLG1CQUFtQjtTQUM5QixDQUFDLENBQUM7UUFFSCxJQUFJLEdBQUcsQ0FBQyxTQUFTLENBQUMsSUFBSSxFQUFFLGVBQWUsRUFBRTtZQUN2QyxLQUFLLEVBQUUsVUFBVSxDQUFDLGFBQWE7U0FDaEMsQ0FBQyxDQUFDO0lBQ0wsQ0FBQztDQUNGO0FBdE1ELHdDQXNNQyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIGNkayBmcm9tICdhd3MtY2RrLWxpYic7XHJcbmltcG9ydCAqIGFzIGVjciBmcm9tICdhd3MtY2RrLWxpYi9hd3MtZWNyJztcclxuaW1wb3J0ICogYXMgZWNzIGZyb20gJ2F3cy1jZGstbGliL2F3cy1lY3MnO1xyXG5pbXBvcnQgKiBhcyBlYzIgZnJvbSAnYXdzLWNkay1saWIvYXdzLWVjMic7XHJcbmltcG9ydCAqIGFzIGVsYnYyIGZyb20gJ2F3cy1jZGstbGliL2F3cy1lbGFzdGljbG9hZGJhbGFuY2luZ3YyJztcclxuaW1wb3J0ICogYXMgaWFtIGZyb20gJ2F3cy1jZGstbGliL2F3cy1pYW0nO1xyXG5pbXBvcnQgKiBhcyBlbGFzdGljYWNoZSBmcm9tICdhd3MtY2RrLWxpYi9hd3MtZWxhc3RpY2FjaGUnO1xyXG5pbXBvcnQgKiBhcyBsb2dzIGZyb20gJ2F3cy1jZGstbGliL2F3cy1sb2dzJztcclxuaW1wb3J0ICogYXMgY2VydGlmaWNhdGVtYW5hZ2VyIGZyb20gJ2F3cy1jZGstbGliL2F3cy1jZXJ0aWZpY2F0ZW1hbmFnZXInO1xyXG5pbXBvcnQgKiBhcyByb3V0ZTUzIGZyb20gJ2F3cy1jZGstbGliL2F3cy1yb3V0ZTUzJztcclxuaW1wb3J0ICogYXMgdGFyZ2V0cyBmcm9tICdhd3MtY2RrLWxpYi9hd3Mtcm91dGU1My10YXJnZXRzJztcclxuaW1wb3J0IHsgQ29uc3RydWN0IH0gZnJvbSAnY29uc3RydWN0cyc7XHJcblxyXG5leHBvcnQgY2xhc3MgQXV0b0F1ZGl0U3RhY2sgZXh0ZW5kcyBjZGsuU3RhY2sge1xyXG4gIGNvbnN0cnVjdG9yKHNjb3BlOiBDb25zdHJ1Y3QsIGlkOiBzdHJpbmcsIHByb3BzPzogY2RrLlN0YWNrUHJvcHMpIHtcclxuICAgIHN1cGVyKHNjb3BlLCBpZCwgcHJvcHMpO1xyXG5cclxuICAgIGNvbnN0IGhvc3RlZFpvbmUgPSByb3V0ZTUzLkhvc3RlZFpvbmUuZnJvbUxvb2t1cCh0aGlzLCAnSG9zdGVkWm9uZScsIHtcclxuICAgICAgZG9tYWluTmFtZTogJ2JpemFpLmVzJyxcclxuICAgIH0pO1xyXG5cclxuICAgIGNvbnN0IGNlcnRpZmljYXRlID0gY2VydGlmaWNhdGVtYW5hZ2VyLkNlcnRpZmljYXRlLmZyb21DZXJ0aWZpY2F0ZUFybihcclxuICAgICAgdGhpcyxcclxuICAgICAgJ0JpemFpQ2VydCcsXHJcbiAgICAgICdhcm46YXdzOmFjbTpldS13ZXN0LTM6NTMzMjY3MTM5NTAzOmNlcnRpZmljYXRlL2MxODVmZjJmLTJmOGItNGNjOC04MTM3LWRmMjA2N2RiMzhmMSdcclxuICAgICk7XHJcblxyXG4gICAgY29uc3QgdnBjID0gbmV3IGVjMi5WcGModGhpcywgJ0F1dG9BdWRpdFZQQycsIHtcclxuICAgICAgbWF4QXpzOiAyLFxyXG4gICAgICBuYXRHYXRld2F5czogMSxcclxuICAgIH0pO1xyXG5cclxuICAgIGNvbnN0IHJlcG9zaXRvcnkgPSBlY3IuUmVwb3NpdG9yeS5mcm9tUmVwb3NpdG9yeU5hbWUodGhpcywgJ0F1dG9BdWRpdFJlcG8nLCAnYXV0by1hdWRpdCcpO1xyXG5cclxuICAgIGNvbnN0IGNsdXN0ZXIgPSBuZXcgZWNzLkNsdXN0ZXIodGhpcywgJ0F1dG9BdWRpdENsdXN0ZXInLCB7XHJcbiAgICAgIHZwYyxcclxuICAgICAgY2x1c3Rlck5hbWU6ICdhdXRvLWF1ZGl0LWNsdXN0ZXInLFxyXG4gICAgfSk7XHJcblxyXG4gICAgY29uc3QgcmVkaXNTdWJuZXRHcm91cCA9IG5ldyBlbGFzdGljYWNoZS5DZm5TdWJuZXRHcm91cCh0aGlzLCAnUmVkaXNTdWJuZXRHcm91cCcsIHtcclxuICAgICAgc3VibmV0SWRzOiB2cGMucHJpdmF0ZVN1Ym5ldHMubWFwKHN1Ym5ldCA9PiBzdWJuZXQuc3VibmV0SWQpLFxyXG4gICAgICBkZXNjcmlwdGlvbjogJ1N1Ym5ldCBncm91cCBmb3IgUmVkaXMgY2FjaGUnLFxyXG4gICAgfSk7XHJcblxyXG4gICAgY29uc3QgcmVkaXNTZWN1cml0eUdyb3VwID0gbmV3IGVjMi5TZWN1cml0eUdyb3VwKHRoaXMsICdSZWRpc1NlY3VyaXR5R3JvdXAnLCB7XHJcbiAgICAgIHZwYyxcclxuICAgICAgZGVzY3JpcHRpb246ICdTZWN1cml0eSBncm91cCBmb3IgUmVkaXMgY2FjaGUnLFxyXG4gICAgICBhbGxvd0FsbE91dGJvdW5kOiB0cnVlLFxyXG4gICAgfSk7XHJcblxyXG4gICAgY29uc3QgcmVkaXMgPSBuZXcgZWxhc3RpY2FjaGUuQ2ZuQ2FjaGVDbHVzdGVyKHRoaXMsICdSZWRpc0NsdXN0ZXInLCB7XHJcbiAgICAgIGVuZ2luZTogJ3JlZGlzJyxcclxuICAgICAgY2FjaGVOb2RlVHlwZTogJ2NhY2hlLnQzLm1pY3JvJyxcclxuICAgICAgbnVtQ2FjaGVOb2RlczogMSxcclxuICAgICAgdnBjU2VjdXJpdHlHcm91cElkczogW3JlZGlzU2VjdXJpdHlHcm91cC5zZWN1cml0eUdyb3VwSWRdLFxyXG4gICAgICBjYWNoZVN1Ym5ldEdyb3VwTmFtZTogcmVkaXNTdWJuZXRHcm91cC5yZWYsXHJcbiAgICB9KTtcclxuXHJcbiAgICBjb25zdCB0YXNrUm9sZSA9IG5ldyBpYW0uUm9sZSh0aGlzLCAnQXV0b0F1ZGl0VGFza1JvbGUnLCB7XHJcbiAgICAgIGFzc3VtZWRCeTogbmV3IGlhbS5TZXJ2aWNlUHJpbmNpcGFsKCdlY3MtdGFza3MuYW1hem9uYXdzLmNvbScpLFxyXG4gICAgfSk7XHJcblxyXG4gICAgdGFza1JvbGUuYWRkTWFuYWdlZFBvbGljeShcclxuICAgICAgaWFtLk1hbmFnZWRQb2xpY3kuZnJvbUF3c01hbmFnZWRQb2xpY3lOYW1lKCdzZXJ2aWNlLXJvbGUvQW1hem9uRUNTVGFza0V4ZWN1dGlvblJvbGVQb2xpY3knKVxyXG4gICAgKTtcclxuXHJcbiAgICBjb25zdCB0YXNrRGVmaW5pdGlvbiA9IG5ldyBlY3MuRmFyZ2F0ZVRhc2tEZWZpbml0aW9uKHRoaXMsICdBdXRvQXVkaXRUYXNrRGVmJywge1xyXG4gICAgICBtZW1vcnlMaW1pdE1pQjogMjA0OCxcclxuICAgICAgY3B1OiAxMDI0LFxyXG4gICAgICB0YXNrUm9sZSxcclxuICAgICAgZmFtaWx5OiAnYXV0by1hdWRpdCcsXHJcbiAgICB9KTtcclxuXHJcbiAgICBjb25zdCBjb250YWluZXIgPSB0YXNrRGVmaW5pdGlvbi5hZGRDb250YWluZXIoJ0F1dG9BdWRpdENvbnRhaW5lcicsIHtcclxuICAgICAgaW1hZ2U6IGVjcy5Db250YWluZXJJbWFnZS5mcm9tRWNyUmVwb3NpdG9yeShyZXBvc2l0b3J5KSxcclxuICAgICAgbWVtb3J5TGltaXRNaUI6IDIwNDgsXHJcbiAgICAgIGxvZ2dpbmc6IGVjcy5Mb2dEcml2ZXJzLmF3c0xvZ3Moe1xyXG4gICAgICAgIHN0cmVhbVByZWZpeDogJ2F1dG8tYXVkaXQnLFxyXG4gICAgICAgIGxvZ1JldGVudGlvbjogbG9ncy5SZXRlbnRpb25EYXlzLk9ORV9NT05USCxcclxuICAgICAgfSksXHJcbiAgICAgIGVudmlyb25tZW50OiB7XHJcbiAgICAgICAgUkVESVNfSE9TVDogcmVkaXMuYXR0clJlZGlzRW5kcG9pbnRBZGRyZXNzLFxyXG4gICAgICAgIFJFRElTX1BPUlQ6IHJlZGlzLmF0dHJSZWRpc0VuZHBvaW50UG9ydCxcclxuICAgICAgfSxcclxuICAgIH0pO1xyXG5cclxuICAgIGNvbnRhaW5lci5hZGRQb3J0TWFwcGluZ3MoXHJcbiAgICAgIHsgY29udGFpbmVyUG9ydDogODAwMCwgcHJvdG9jb2w6IGVjcy5Qcm90b2NvbC5UQ1AgfSxcclxuICAgICAgeyBjb250YWluZXJQb3J0OiA4NTAxLCBwcm90b2NvbDogZWNzLlByb3RvY29sLlRDUCB9XHJcbiAgICApO1xyXG5cclxuICAgIGNvbnN0IHNlcnZpY2VTZyA9IG5ldyBlYzIuU2VjdXJpdHlHcm91cCh0aGlzLCAnU2VydmljZVNlY3VyaXR5R3JvdXAnLCB7XHJcbiAgICAgIHZwYyxcclxuICAgICAgZGVzY3JpcHRpb246ICdTZWN1cml0eSBncm91cCBmb3IgQXV0byBBdWRpdCBTZXJ2aWNlJyxcclxuICAgICAgYWxsb3dBbGxPdXRib3VuZDogdHJ1ZSxcclxuICAgIH0pO1xyXG5cclxuICAgIC8vIEFsbG93IGluYm91bmQgdHJhZmZpYyBmcm9tIEFMQiB0byBFQ1Mgc2VydmljZVxyXG4gICAgc2VydmljZVNnLmFkZEluZ3Jlc3NSdWxlKFxyXG4gICAgICBzZXJ2aWNlU2csXHJcbiAgICAgIGVjMi5Qb3J0LnRjcCg4MDAwKSxcclxuICAgICAgJ0FsbG93IGluYm91bmQgdHJhZmZpYyB0byBBUEknXHJcbiAgICApO1xyXG5cclxuICAgIHNlcnZpY2VTZy5hZGRJbmdyZXNzUnVsZShcclxuICAgICAgc2VydmljZVNnLFxyXG4gICAgICBlYzIuUG9ydC50Y3AoODUwMSksXHJcbiAgICAgICdBbGxvdyBpbmJvdW5kIHRyYWZmaWMgdG8gU3RyZWFtbGl0J1xyXG4gICAgKTtcclxuXHJcbiAgICBjb25zdCBzZXJ2aWNlID0gbmV3IGVjcy5GYXJnYXRlU2VydmljZSh0aGlzLCAnQXV0b0F1ZGl0U2VydmljZScsIHtcclxuICAgICAgY2x1c3RlcixcclxuICAgICAgdGFza0RlZmluaXRpb24sXHJcbiAgICAgIGRlc2lyZWRDb3VudDogMSxcclxuICAgICAgYXNzaWduUHVibGljSXA6IGZhbHNlLFxyXG4gICAgICBzZWN1cml0eUdyb3VwczogW3NlcnZpY2VTZ10sXHJcbiAgICAgIHZwY1N1Ym5ldHM6IHtcclxuICAgICAgICBzdWJuZXRUeXBlOiBlYzIuU3VibmV0VHlwZS5QUklWQVRFX1dJVEhfRUdSRVNTXHJcbiAgICAgIH1cclxuICAgIH0pO1xyXG5cclxuICAgIGNvbnN0IGxiID0gbmV3IGVsYnYyLkFwcGxpY2F0aW9uTG9hZEJhbGFuY2VyKHRoaXMsICdBdXRvQXVkaXRBTEInLCB7XHJcbiAgICAgIHZwYyxcclxuICAgICAgaW50ZXJuZXRGYWNpbmc6IHRydWUsXHJcbiAgICAgIHNlY3VyaXR5R3JvdXA6IHNlcnZpY2VTZyxcclxuICAgIH0pO1xyXG5cclxuICAgIC8vIENyZWF0ZSBzaW5nbGUgSFRUUFMgbGlzdGVuZXJcclxuICAgIGNvbnN0IGh0dHBzTGlzdGVuZXIgPSBsYi5hZGRMaXN0ZW5lcignSHR0cHNMaXN0ZW5lcicsIHtcclxuICAgICAgcG9ydDogNDQzLFxyXG4gICAgICBwcm90b2NvbDogZWxidjIuQXBwbGljYXRpb25Qcm90b2NvbC5IVFRQUyxcclxuICAgICAgY2VydGlmaWNhdGVzOiBbY2VydGlmaWNhdGVdLFxyXG4gICAgICBkZWZhdWx0QWN0aW9uOiBlbGJ2Mi5MaXN0ZW5lckFjdGlvbi5maXhlZFJlc3BvbnNlKDQwNCksXHJcbiAgICB9KTtcclxuXHJcbiAgICAvLyBBZGQgQVBJIHRhcmdldCBncm91cFxyXG4gICAgY29uc3QgYXBpVGFyZ2V0R3JvdXAgPSBodHRwc0xpc3RlbmVyLmFkZFRhcmdldHMoJ0FwaVRhcmdldCcsIHtcclxuICAgICAgcG9ydDogODAwMCxcclxuICAgICAgcHJvdG9jb2w6IGVsYnYyLkFwcGxpY2F0aW9uUHJvdG9jb2wuSFRUUCxcclxuICAgICAgdGFyZ2V0czogW1xyXG4gICAgICAgIHNlcnZpY2UubG9hZEJhbGFuY2VyVGFyZ2V0KHtcclxuICAgICAgICAgIGNvbnRhaW5lck5hbWU6ICdBdXRvQXVkaXRDb250YWluZXInLFxyXG4gICAgICAgICAgY29udGFpbmVyUG9ydDogODAwMCxcclxuICAgICAgICB9KSxcclxuICAgICAgXSxcclxuICAgICAgY29uZGl0aW9uczogW1xyXG4gICAgICAgIGVsYnYyLkxpc3RlbmVyQ29uZGl0aW9uLmhvc3RIZWFkZXJzKFsnYXBpLmJpemFpLmVzJ10pLFxyXG4gICAgICBdLFxyXG4gICAgICBwcmlvcml0eTogMSxcclxuICAgICAgaGVhbHRoQ2hlY2s6IHtcclxuICAgICAgICBwYXRoOiAnL2hlYWx0aCcsXHJcbiAgICAgICAgdW5oZWFsdGh5VGhyZXNob2xkQ291bnQ6IDIsXHJcbiAgICAgICAgaGVhbHRoeVRocmVzaG9sZENvdW50OiA1LFxyXG4gICAgICAgIGludGVydmFsOiBjZGsuRHVyYXRpb24uc2Vjb25kcygzMCksXHJcbiAgICAgIH0sXHJcbiAgICB9KTtcclxuXHJcbiAgICAvLyBBZGQgU3RyZWFtbGl0IHRhcmdldCBncm91cFxyXG4gICAgY29uc3Qgc3RyZWFtbGl0VGFyZ2V0R3JvdXAgPSBodHRwc0xpc3RlbmVyLmFkZFRhcmdldHMoJ1N0cmVhbWxpdFRhcmdldCcsIHtcclxuICAgICAgcG9ydDogODUwMSxcclxuICAgICAgcHJvdG9jb2w6IGVsYnYyLkFwcGxpY2F0aW9uUHJvdG9jb2wuSFRUUCxcclxuICAgICAgdGFyZ2V0czogW1xyXG4gICAgICAgIHNlcnZpY2UubG9hZEJhbGFuY2VyVGFyZ2V0KHtcclxuICAgICAgICAgIGNvbnRhaW5lck5hbWU6ICdBdXRvQXVkaXRDb250YWluZXInLFxyXG4gICAgICAgICAgY29udGFpbmVyUG9ydDogODUwMSxcclxuICAgICAgICB9KSxcclxuICAgICAgXSxcclxuICAgICAgY29uZGl0aW9uczogW1xyXG4gICAgICAgIGVsYnYyLkxpc3RlbmVyQ29uZGl0aW9uLmhvc3RIZWFkZXJzKFsnYml6YWkuZXMnXSksXHJcbiAgICAgIF0sXHJcbiAgICAgIHByaW9yaXR5OiAyLFxyXG4gICAgICBoZWFsdGhDaGVjazoge1xyXG4gICAgICAgIHBhdGg6ICcvJyxcclxuICAgICAgICB1bmhlYWx0aHlUaHJlc2hvbGRDb3VudDogMixcclxuICAgICAgICBoZWFsdGh5VGhyZXNob2xkQ291bnQ6IDUsXHJcbiAgICAgICAgaW50ZXJ2YWw6IGNkay5EdXJhdGlvbi5zZWNvbmRzKDMwKSxcclxuICAgICAgfSxcclxuICAgIH0pO1xyXG5cclxuICAgIC8vIENyZWF0ZSBETlMgcmVjb3JkcyBmb3IgYm90aCBzZXJ2aWNlc1xyXG4gICAgbmV3IHJvdXRlNTMuQVJlY29yZCh0aGlzLCAnQXBpQWxpYXNSZWNvcmQnLCB7XHJcbiAgICAgIHpvbmU6IGhvc3RlZFpvbmUsXHJcbiAgICAgIHJlY29yZE5hbWU6ICdhcGknLFxyXG4gICAgICB0YXJnZXQ6IHJvdXRlNTMuUmVjb3JkVGFyZ2V0LmZyb21BbGlhcyhuZXcgdGFyZ2V0cy5Mb2FkQmFsYW5jZXJUYXJnZXQobGIpKSxcclxuICAgIH0pO1xyXG5cclxuICAgIG5ldyByb3V0ZTUzLkFSZWNvcmQodGhpcywgJ0FwcEFsaWFzUmVjb3JkJywge1xyXG4gICAgICB6b25lOiBob3N0ZWRab25lLFxyXG4gICAgICByZWNvcmROYW1lOiAnYXBwJyxcclxuICAgICAgdGFyZ2V0OiByb3V0ZTUzLlJlY29yZFRhcmdldC5mcm9tQWxpYXMobmV3IHRhcmdldHMuTG9hZEJhbGFuY2VyVGFyZ2V0KGxiKSksXHJcbiAgICB9KTtcclxuXHJcbiAgICAvLyBPdXRwdXQgdGhlIFVSTHMgZm9yIGVhc3kgYWNjZXNzXHJcbiAgICBuZXcgY2RrLkNmbk91dHB1dCh0aGlzLCAnU3RyZWFtbGl0VVJMJywge1xyXG4gICAgICB2YWx1ZTogJ2h0dHBzOi8vYml6YWkuZXMnLFxyXG4gICAgICBkZXNjcmlwdGlvbjogJ1VSTCBmb3IgdGhlIFN0cmVhbWxpdCBhcHBsaWNhdGlvbicsXHJcbiAgICB9KTtcclxuXHJcbiAgICBuZXcgY2RrLkNmbk91dHB1dCh0aGlzLCAnQXBpVVJMJywge1xyXG4gICAgICB2YWx1ZTogJ2h0dHBzOi8vYXBpLmJpemFpLmVzJyxcclxuICAgICAgZGVzY3JpcHRpb246ICdVUkwgZm9yIHRoZSBBUEknLFxyXG4gICAgfSk7XHJcblxyXG4gICAgbmV3IGNkay5DZm5PdXRwdXQodGhpcywgJ0xvYWRCYWxhbmNlckROUycsIHtcclxuICAgICAgdmFsdWU6IGxiLmxvYWRCYWxhbmNlckRuc05hbWUsXHJcbiAgICB9KTtcclxuXHJcbiAgICBuZXcgY2RrLkNmbk91dHB1dCh0aGlzLCAnUmVwb3NpdG9yeVVSSScsIHtcclxuICAgICAgdmFsdWU6IHJlcG9zaXRvcnkucmVwb3NpdG9yeVVyaSxcclxuICAgIH0pO1xyXG4gIH1cclxufVxyXG4iXX0=