#!/bin/bash -e

python -m envs.griddly.enjoy \
    --env=GDY-Forage \
    --device=cpu \
    --train_dir=./train_dir/ \
    --fps=10 \
    --max_num_frames=1000 \
    --eval_env_frameskip=1 \
    --experiment "$@"

