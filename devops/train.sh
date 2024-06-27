#/bin/bash -e

pkill -9 -f wandb
pkill -9 -f python
git pull
python -m tools.run \
    framework=sample_factory/train/prod \
    wandb.track=true \
    "$@"
