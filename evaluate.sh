#!/bin/bash -e

python -m framework.sample_factory_framework.evaluate \
    --seed=0 \
    --device=cpu \
    --train_dir=./train_dir/ \
    --max_num_episodes=1 \
    --eval_env_frameskip=1 \
    --load_checkpoint_kind=latest \
   "$@"

