import boto3
import argparse
import netrc
import os
import logging

def launch_task(args, task_args):
    ecs = boto3.client('ecs')

    # Get the latest version of the task definition
    response = ecs.describe_task_definition(taskDefinition=args.task_def)
    task_definition = response['taskDefinition']['taskDefinitionArn']

    # Get the wandb key from the .netrc file
    netrc_info = netrc.netrc(os.path.expanduser('~/.netrc'))
    wandb_key = netrc_info.authenticators('api.wandb.ai')[2]
    if not wandb_key:
        raise ValueError('WANDB_API_KEY not found in .netrc file')

    setup_cmds = [
        'git pull',
        'ln -s /mnt/efs/train_dir train_dir',
    ]
    train_cmd = [
        './trainers/a100_100x100_simple.sh',
        f'--experiment={args.experiment}',
        '--batch_size=4096',
        *task_args,
    ]
    if args.git_branch is not None:
        setup_cmds.append(f'git checkout {args.git_branch}')
    if args.init_model is not None:
        setup_cmds.append(f'./devops/load_model.sh {args.init_model}',)
        train_cmd.append(f'--init_checkpoint_path=train_dir/{args.init_model}/latest.pth')
    if args.num_workers is not None:
        train_cmd.append(f'--num_workers={args.num_workers}')

    print("Setup commands:", "\n".join(setup_cmds))
    print("Train command:", " ".join(train_cmd))
    overrides = {
        'containerOverrides': [
            {
                'name': 'metta',
                'command': ["; ".join([
                    *setup_cmds,
                    " ".join(train_cmd),
                ])],
                'environment': [
                    {
                        'name': 'WANDB_API_KEY',
                        'value': wandb_key
                    },
                ]
            }
        ]
    }

    print(f"launching task using {task_definition}")

    # Launch the task
    response = ecs.run_task(
        cluster=args.cluster,
        taskDefinition=task_definition,
        startedBy=args.experiment,
        launchType='EC2',
        overrides=overrides,
        # networkConfiguration = {
        #     'awsvpcConfiguration': {
        #         'subnets': [
        #             'subnet-07d3d46baf6dc2016',
        #             'subnet-08e4e4c89aaa460da',
        #             'subnet-015146d729b05250d',
        #             'subnet-013868c597718f825',
        #             'subnet-0a4eeb61451368b33',
        #             'subnet-0e882abc5155fdc12'],
        #         'securityGroups': [
        #             'sg-037e6e08491e40608', # defqult
        #             'sg-04abcd27b35bda23b' # metta-ecs-sg
        #         ],
        #     }
        # },
    )

    if response['ResponseMetadata']['HTTPStatusCode'] == 200 and response['tasks']:
        task_id = response['tasks'][0]['taskArn']
        print(f'Task submitted: {task_id}')
        print(
            "https://us-east-1.console.aws.amazon.com/ecs/v2/clusters/" +
            args.cluster +
            "/tasks/" +
            task_id.split('/')[-1] +
            "/?selectedContainer=metta")
    else:
        logging.error('Failed to submit: %s', response)

    def submit_batch_job(args, setup_cmds, train_cmd, wandb_key):
        batch = boto3.client('batch')

        job_name = f"{args.experiment}"
        job_queue = args.cluster
        job_definition = "my_job_definition"  # replace with your job definition

        response = batch.submit_job(
            jobName=job_name,
            jobQueue=job_queue,
            jobDefinition=job_definition,
            containerOverrides={
                'command': ["; ".join([
                    *setup_cmds,
                    " ".join(train_cmd),
                ])],
                'environment': [
                    {
                        'name': 'WANDB_API_KEY',
                        'value': wandb_key
                    },
                ]
            }
        )

        print(f"Submitted job {job_name} to queue {job_queue} with job ID {response['jobId']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch an ECS task with a wandb key.')
    parser.add_argument('--cluster', default="metta", help='The name of the ECS cluster.')
    parser.add_argument('--task-def', default="metta-trainer", help='The family or ARN of the task definition.')
    parser.add_argument('--experiment', required=True, help='The experiment to run.')
    parser.add_argument('--init_model', default=None, help='The experiment to run.')
    parser.add_argument('--num_workers', default=None, type=int, help='Number of rollout workers')
    parser.add_argument('--git_branch', default=None, help='The git branch to use for the task.')

    args, task_args = parser.parse_known_args()


    launch_task(args, task_args)
