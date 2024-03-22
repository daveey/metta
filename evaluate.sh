#!/bin/bash -e

python -m enjoy \
    --seed=0 \
    --env=GDY-PowerGrid \
    --device=cpu \
    --train_dir=./train_dir/ \
    --max_num_episodes=1 \
    --eval_env_frameskip=1 \
    --load_checkpoint_kind=latest \
   "$@"

