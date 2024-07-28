#/bin/bash -e

pkill -9 -f wandb
pkill -9 -f python
git pull
python setup.py build_ext --inplace
python -m tools.run \
    framework=sample_factory \
    cmd=train \
    hardware=pufferbox \
    wandb.track=true \
    "$@"
