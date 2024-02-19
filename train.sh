#/bin/bash -e

python -m envs.griddly.train \
    --algo=APPO \
    --env=GDY-PowerGrid \
    --with_wandb=True \
    --num_workers=25 \
    --decorrelate_experience_max_seconds=100 \
    --env_num_agents=20 \
    --power_grid_width=20:100 \
    --power_grid_height=20:100 \
    --power_grid_wall_density=0.1 \
    --power_grid_chargers_per_agent=1:10 \
    --power_grid_initial_energy=50 \
    "$@"
