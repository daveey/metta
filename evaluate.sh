#!/bin/bash -e

python -m framework.sample_factory.evaluate \
    --seed=0 \
    --config=evaluation \
    --config=env_a5_25x25 \
   "$@"

