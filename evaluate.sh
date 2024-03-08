#!/bin/bash -e

python -m enjoy \
    --seed=0 \
    --env=GDY-PowerGrid \
    --device=cpu \
    --train_dir=./train_dir/ \
    --fps=5 \
    --max_num_frames=1000 \
    --eval_env_frameskip=1 \
    --load_checkpoint_kind=latest \
   "$@"

