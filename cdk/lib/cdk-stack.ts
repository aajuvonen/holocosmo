import * as cdk from "aws-cdk-lib";
import { Bucket } from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";
import * as batch from "aws-cdk-lib/aws-batch";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as iam from "aws-cdk-lib/aws-iam";
import { ContainerImage } from "aws-cdk-lib/aws-ecs";
import { Platform } from "aws-cdk-lib/aws-ecr-assets";

export class CdkStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const bucket = new Bucket(this, "ArtifactBucket", {
      bucketName: "holocosmo-artifacts",
    });

    // Create a VPC for the Batch compute environment.
    const vpc = ec2.Vpc.fromLookup(this, "Vpc", { isDefault: true });

    const jobRole = new iam.Role(this, "JobRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
    });
    jobRole.attachInlinePolicy(
      new iam.Policy(this, "LogsPolicy", {
        statements: [
          new iam.PolicyStatement({
            resources: ["*"],
            actions: [
              "logs:CreateLogGroup",
              "logs:CreateLogStream",
              "logs:PutLogEvents",
            ],
          }),
        ],
      })
    );

    const computeEnvironment = new batch.ManagedEc2EcsComputeEnvironment(
      this,
      "CE",
      {
        vpc,
        // Uncomment to enable more EBS storage allocation
        // launchTemplate: new ec2.LaunchTemplate(this, "LaunchTemplateCE2", {
        //   blockDevices: [
        //     {
        //       deviceName: "/dev/xvda",
        //       volume: ec2.BlockDeviceVolume.ebs(1024), // 1TB
        //     },
        //   ],
        // }),
        instanceClasses: [
          ec2.InstanceClass.C6G,
          ec2.InstanceClass.C7G,
          ec2.InstanceClass.M6G,
          ec2.InstanceClass.M7G,
          ec2.InstanceClass.R6G,
          ec2.InstanceClass.R7G,
        ],
        useOptimalInstanceClasses: false,
        replaceComputeEnvironment: true,
      }
    );

    const queue = new batch.JobQueue(this, "Queue", {
      computeEnvironments: [
        {
          computeEnvironment,
          order: 1,
        },
      ],
    });

    const batchJob = new batch.EcsJobDefinition(this, "BatchJob", {
      jobDefinitionName: "holocosmo-batchjob",
      timeout: cdk.Duration.days(1),
      container: new batch.EcsEc2ContainerDefinition(this, "Container", {
        image: ContainerImage.fromAsset("../path-to-docker-dir", {
          // platform: Platform.LINUX_ARM64,
        }),
        cpu: 64,
        memory: cdk.Size.gibibytes(248),
        jobRole,
        command: [
          // arguments
          // "Ref::foo",
          // "Ref::bar",
          // "Ref::baz",
        ],
        volumes: [
          batch.EcsVolume.host({
            name: "nvme-disk",
            hostPath: "/nvme",
            containerPath: "/nvme",
          }),
        ],
        environment: {
          BUCKET: bucket.bucketName,
        },
      }),
      // default params to arguments
      // parameters: {
      //   foo: "foo",
      //   bar: "bar",
      //   baz: "baz",
      // },
    });

    // Grant job right to read/write on artifact bucket
    bucket.grantReadWrite(jobRole);
  }
}
