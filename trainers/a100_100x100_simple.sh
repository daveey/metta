#/bin/bash -e


./train.sh \
    --env_width=100 \
    --env_height=100 \
    --env_num_agents=100 \
    --env_num_altars=20 \
    --env_num_chargers=30 \
    --env_num_generators=200 \
    --env_wall_density=0.01 \
    --env_reward_rank_steps=1000 \
    --env_altar:cost=100 \
    --env_reward_prestige_weight=0 \
    --env_cost:shield=0 \
    --max_policy_lag=20000 \
    "$@"
