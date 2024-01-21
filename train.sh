#/bin/bash -e

python -m envs.griddly.train \
    --algo=APPO \
    --env=GDY-GDY-Forage \
    --with_wandb=True \
    --num_workers=25 \
    --decorrelate_experience_max_seconds=100 \
    --lr_schedule=kl_adaptive_epoch \
    "$@"

