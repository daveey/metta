#/bin/bash -e

python -m envs.griddly.train \
    --algo=APPO \
    --env=GDY-PowerGrid \
    --with_wandb=True \
    --num_workers=25 \
    --decorrelate_experience_max_seconds=100 \
    --env_num_agents=20 \
    --env_width=20:100 \
    --env_height=20:100 \
    --env_chargers_per_agent=1:10 \
    "$@"
