#/bin/bash -e

./experiments/train.sh \
    --env_num_agents=2 \
    "$@"
