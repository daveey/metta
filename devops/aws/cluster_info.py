import boto3
import argparse

def get_public_ips(cluster):
    ecs = boto3.client('ecs')
    ec2 = boto3.client('ec2')

    # List the container instances in the cluster
    response = ecs.list_container_instances(cluster=cluster)
    container_instance_arns = response['containerInstanceArns']

    # Describe the container instances to get the EC2 instance IDs
    response = ecs.describe_container_instances(cluster=cluster, containerInstances=container_instance_arns)
    ec2_instance_ids = [ci['ec2InstanceId'] for ci in response['containerInstances']]

    # Describe the EC2 instances to get the public IPs
    response = ec2.describe_instances(InstanceIds=ec2_instance_ids)
    public_ips = [instance['PublicIpAddress'] for reservation in response['Reservations'] for instance in reservation['Instances']]

    return public_ips

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the public IPs of all the container instances in an ECS cluster.')
    parser.add_argument('--cluster', required=True, help='The name of the ECS cluster.')
    args = parser.parse_args()

    public_ips = get_public_ips(args.cluster)
    print(public_ips)
