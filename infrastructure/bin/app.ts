#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { AutoAuditStack } from '../lib/auto-audit-stack';

const app = new cdk.App();
new AutoAuditStack(app, 'AutoAuditStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || 'eu-west-3',
  },
}); 