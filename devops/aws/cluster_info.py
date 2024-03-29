import boto3
import argparse
from colorama import init, Fore, Style

def get_batch_job_queues():
    batch = boto3.client('batch')
    response = batch.describe_job_queues()
    return [queue['jobQueueName'] for queue in response['jobQueues']]

def get_batch_jobs(job_queue, job_status, max_jobs):
    batch = boto3.client('batch')
    kwargs = {'jobQueue': job_queue, 'maxResults': max_jobs}
    if job_status:
        kwargs['jobStatus'] = job_status
    response = batch.list_jobs(**kwargs)
    jobs = response['jobSummaryList']

    job_details = []
    for job in jobs:
        job_id = job['jobId']
        job_name = job['jobName']
        job_status = job['status']
        job_link = f"https://console.aws.amazon.com/batch/home?region=us-east-1#jobs/detail/{job_id}"

        # Get the EC2 instance ID and public IP
        job_desc = batch.describe_jobs(jobs=[job_id])
        container = job_desc['jobs'][0]['container']
        instance_id = container.get('ec2InstanceId')

        public_ip = ''
        if instance_id:
            ec2 = boto3.client('ec2')
            instances = ec2.describe_instances(InstanceIds=[instance_id])
            public_ip = instances['Reservations'][0]['Instances'][0].get('PublicIpAddress', '')

        job_details.append({
            'name': job_name,
            'status': job_status,
            'link': job_link,
            'public_ip': public_ip
        })

    return job_details

def get_ecs_clusters():
    ecs = boto3.client('ecs')
    response = ecs.list_clusters()
    return response['clusterArns']

def get_ecs_tasks(clusters, max_tasks):
    ecs = boto3.client('ecs')

    task_details = []
    for cluster in clusters:
        response = ecs.list_tasks(cluster=cluster, maxResults=max_tasks)
        task_arns = response['taskArns']

        for task_arn in task_arns:
            task_desc = ecs.describe_tasks(cluster=cluster, tasks=[task_arn])
            task = task_desc['tasks'][0]
            task_id = task['taskArn'].split('/')[-1]
            task_name = task['overrides']['containerOverrides'][0]['name']
            task_status = task['lastStatus']
            task_link = f"https://console.aws.amazon.com/ecs/home?region=us-east-1#/clusters/{cluster}/tasks/{task_id}/details"

            # Get the EC2 instance ID and public IP
            container_instance_arn = task['containerInstanceArn']
            container_instance_desc = ecs.describe_container_instances(cluster=cluster, containerInstances=[container_instance_arn])
            ec2_instance_id = container_instance_desc['containerInstances'][0]['ec2InstanceId']

            ec2 = boto3.client('ec2')
            instances = ec2.describe_instances(InstanceIds=[ec2_instance_id])
            public_ip = instances['Reservations'][0]['Instances'][0].get('PublicIpAddress', '')

            task_details.append({
                'name': task_name,
                'status': task_status,
                'link': task_link,
                'public_ip': public_ip
            })

    return task_details

def print_row(key, value, use_color):
    if use_color:
        print(f"  {Fore.BLUE}{key}:{Style.RESET_ALL} {value}")
    else:
        print(f"  {key}: {value}")

def print_status(jobs_by_queue, tasks, use_color):
    for job_queue, jobs in jobs_by_queue.items():
        if use_color:
            print(f"{Fore.CYAN}AWS Batch Jobs - Queue: {job_queue}{Style.RESET_ALL}")
        else:
            print(f"AWS Batch Jobs - Queue: {job_queue}")

        for job in jobs:
            status_color = Fore.GREEN if job['status'] == 'SUCCEEDED' else Fore.YELLOW if job['status'] == 'RUNNING' else Fore.RED
            print_row("Name", job['name'], use_color)
            print_row("Status", f"{status_color}{job['status']}{Style.RESET_ALL}" if use_color else job['status'], use_color)
            print_row("Link", job['link'], use_color)
            print_row("Public IP", job['public_ip'], use_color)
            print()

    if tasks:
        if use_color:
            print(f"{Fore.CYAN}ECS Tasks{Style.RESET_ALL}")
        else:
            print("ECS Tasks")

        for task in tasks:
            status_color = Fore.GREEN if task['status'] == 'RUNNING' else Fore.RED
            print_row("Name", task['name'], use_color)
            print_row("Status", f"{status_color}{task['status']}{Style.RESET_ALL}" if use_color else task['status'], use_color)
            print_row("Link", task['link'], use_color)
            print_row("Public IP", task['public_ip'], use_color)
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the status of AWS Batch jobs and ECS tasks.')
    parser.add_argument('--max-jobs', type=int, default=10, help='The maximum number of jobs to display.')
    parser.add_argument('--ecs', action='store_true', help='Include ECS tasks in the status dump.')
    parser.add_argument('--no-color', action='store_true', help='Disable color output.')
    parser.add_argument('--running', action='store_true', help='Show only running jobs.')
    parser.add_argument('--failed', action='store_true', help='Show only failed jobs.')
    args = parser.parse_args()

    init()  # Initialize colorama

    job_status = 'RUNNING' if args.running else 'FAILED' if args.failed else None

    job_queues = get_batch_job_queues()
    jobs_by_queue = {queue: get_batch_jobs(queue, job_status, args.max_jobs) for queue in job_queues}

    ecs_tasks = []
    if args.ecs:
        ecs_clusters = get_ecs_clusters()
        ecs_tasks = get_ecs_tasks(ecs_clusters, args.max_jobs)

    print_status(jobs_by_queue, ecs_tasks, not args.no_color)
