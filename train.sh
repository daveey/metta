#/bin/bash -e

python -m envs.griddly.train \
    --algo=APPO \
    --env=GDY-OrbWorld \
    --with_wandb=True \
    --num_workers=25 \
    --decorrelate_experience_max_seconds=100 \
    --env_num_agents=50 \
    --orb_world_width=30:200 \
    --orb_world_height=30:200 \
    --orb_world_wall_density=0.1 \
    --orb_world_factories_per_agent=1:10 \
    --orb_world_initial_energy=50 \
    "$@"
