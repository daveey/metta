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
    container_instances = {ci['containerInstanceArn']: {'ec2InstanceId': ci['ec2InstanceId'], 'tasks': []} for ci in response['containerInstances']}

    # Describe the EC2 instances to get the public IPs
    ec2_instances = ec2.describe_instances(InstanceIds=[ci['ec2InstanceId'] for ci in container_instances.values()])

    # List the tasks in the cluster
    response = ecs.list_tasks(cluster=cluster)
    task_arns = response['taskArns']

    # Describe the tasks to get the launchedBy attribute
    response = ecs.describe_tasks(cluster=cluster, tasks=task_arns)
    for task in response['tasks']:
        container_instances[task['containerInstanceArn']]['tasks'].append(task)

    return ec2_instances, container_instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the public IPs of all the container instances in an ECS cluster.')
    parser.add_argument('--cluster', required=True, help='The name of the ECS cluster.')
    args = parser.parse_args()

    res, container_instances = get_container_instances(args.cluster)
    for reservation in res['Reservations']:
        for instance in reservation['Instances']:
            for ci in container_instances.values():
                if ci['ec2InstanceId'] == instance['InstanceId']:
                    for task in ci['tasks']:
                        print(f"{task['startedBy']}: alias tsh='ssh -i ~/.ssh/aws.pem ec2-user@{instance['PublicIpAddress']}'")
                    if not ci['tasks']:
                        print(f"No tasks: alias tsh='ssh -i ~/.ssh/aws.pem ec2-user@{instance['PublicIpAddress']}'")
