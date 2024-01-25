#/bin/bash -e

python -m envs.griddly.train \
    --algo=APPO \
    --env=GDY-Forage \
    --with_wandb=True \
    --num_workers=25 \
    --decorrelate_experience_max_seconds=100 \
    --lr_schedule=kl_adaptive_epoch \
    --forage.num_agents=10 \
    --forage.width_min=10 \
    --forage.width_max=100 \
    --forage.height_min=10 \
    --forage.height_max=100 \
    --forage.energy_per_agent=2 \
    --forage.wall_density=0.1 \
    "$@"

