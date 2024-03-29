#/bin/bash -e

pkill -9 wandb
pkill -9 python
git pull
python -m framework.sample_factory.train \
    --config=training \
    --config=env_a5_25x25 \
    --config=training \
    "$@"
