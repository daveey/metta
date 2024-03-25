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

    setup_cmds = [
        'git pull',
    ]
    train_cmd = [
        './trainers/a100_100x100_simple.sh',
        f'--experiment={args.experiment}',
        '--batch_size=4096',
    ]
    if args.init_model is not None:
        setup_cmds.append(f'./devops/load_model.sh {args.init_model}',)
        train_cmd.append(f'--init_checkpoint_path=train_dir/{args.init_model}/latest.pth')
    if args.num_workers is not None:
        train_cmd.append(f'--num_workers={args.num_workers}')

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch an ECS task with a wandb key.')
    parser.add_argument('--cluster', default="metta", help='The name of the ECS cluster.')
    parser.add_argument('--task-def', default="metta-trainer", help='The family or ARN of the task definition.')
    parser.add_argument('--experiment', required=True, help='The experiment to run.')
    parser.add_argument('--init_model', default=None, help='The experiment to run.')
    parser.add_argument('--num_workers', default=None, type=int, help='Number of rollout workers')
    args = parser.parse_args()

    launch_task(args)
