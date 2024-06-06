#/bin/bash -e

pkill -9 -f wandb
pkill -9 -f python
git pull
python -m tools.train \
    +sample_factory=train_prod \
    "$@"
