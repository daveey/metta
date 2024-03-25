import boto3
import argparse
import netrc
import os
import logging

def launch_task(args):
    ecs = boto3.client('ecs')

    # Get the latest version of the task definition
    response = ecs.describe_task_definition(taskDefinition=args.task_def)
    task_definition = response['taskDefinition']['taskDefinitionArn']

    # Get the wandb key from the .netrc file
    netrc_info = netrc.netrc(os.path.expanduser('~/.netrc'))
    wandb_key = netrc_info.authenticators('api.wandb.ai')[2]
    if not wandb_key:
        raise ValueError('WANDB_API_KEY not found in .netrc file')

    # Set up the environment variable override for the wandb key
    overrides = {
        'containerOverrides': [
            {
                'name': 'metta',
                'command': ["; ".join([
                    'git pull',
                    f'./devops/load_model.sh {args.init_model}',

                    " ".join([
                        './trainers/a100_100x100_simple.sh',
                        f'--experiment={args.experiment}',
                        '--batch_size=4096',
                        '--num_workers=8',
                        f'--init_checkpoint_path=train_dir/{args.init_model}/latest.pth',
                    ])
                ])],
                'environment': [
                    {
                        'name': 'WANDB_API_KEY',
                        'value': wandb_key
                    }
                ]
            }
        ]
    }

    print(f"launching task using {task_definition}")

    # Launch the task
    response = ecs.run_task(
        cluster=args.cluster,
        taskDefinition=task_definition,
        launchType='EC2',
        overrides=overrides
    )

    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print('Task submitted.')
    else:
        logging.error('Failed to submit: %s', response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch an ECS task with a wandb key.')
    parser.add_argument('--cluster', default="metta", help='The name of the ECS cluster.')
    parser.add_argument('--task-def', default="metta-trainer", help='The family or ARN of the task definition.')
    parser.add_argument('--experiment', required=True, help='The experiment to run.')
    args = parser.parse_args()

    launch_task(args)
