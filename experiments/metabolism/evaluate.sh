#!/bin/bash -e

./experiments/evaluate.sh \
    --env_width=10 \
    --env_height=10 \
    --env_num_altars=1 \
    --env_num_chargers=1 \
    --env_num_generators=10 \
    --env_wall_density=0 \
    --env_cost:shield=10000 \
    --env_cost:attack=10000 \
    --env_agent:initial_energy=0 \
    "$@"
