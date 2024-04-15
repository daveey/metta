#/bin/bash -e

pkill -9 wandb
pkill -9 python
git pull
python -m tools.train \
    +sample_factory=train_prod \
    "$@"
