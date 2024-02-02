#/bin/bash -e

python -m envs.griddly.train \
    --algo=APPO \
    --env=GDY-Forage \
    --with_wandb=True \
    --num_workers=25 \
    --decorrelate_experience_max_seconds=100 \
    --forage_num_agents=10 \
    --forage_width_min=10 \
    --forage_width_max=100 \
    --forage_height_min=10 \
    --forage_height_max=100 \
    --forage_energy_per_agent=1 \
    --forage_wall_density=0.1 \
    "$@"
