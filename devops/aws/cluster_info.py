import boto3
import argparse

def get_container_instances(cluster):
    ecs = boto3.client('ecs')
    ec2 = boto3.client('ec2')

    # List the container instances in the cluster
    response = ecs.list_container_instances(cluster=cluster)
    container_instance_arns = response['containerInstanceArns']

    # Describe the container instances to get the EC2 instance IDs
    response = ecs.describe_container_instances(cluster=cluster, containerInstances=container_instance_arns)
    ec2_instance_ids = [ci['ec2InstanceId'] for ci in response['containerInstances']]

    # Describe the EC2 instances to get the public IPs
    return ec2.describe_instances(InstanceIds=ec2_instance_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the public IPs of all the container instances in an ECS cluster.')
    parser.add_argument('--cluster', required=True, help='The name of the ECS cluster.')
    args = parser.parse_args()

    res = get_container_instances(args.cluster)
    for reservation in res['Reservations']:
        for instance in reservation['Instances']:
            ip = instance['PublicIpAddress']
            print(f"alias tsh='ssh -i ~/.ssh/aws.pem ec2-user@{ip}'")
