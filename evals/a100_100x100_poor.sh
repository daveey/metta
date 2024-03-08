./evaluate.sh \
    --env_width=100 \
    --env_height=100 \
    --env_wall_density=0.01 \
    --env_num_agents=100 \
    --env_num_altars=5 \
    --env_num_chargers=5 \
    --env_num_generators=30 \
    --env_reward_rank_steps=1000 \
    --env_altar:reward=100 \
    --env_altar:cost=100 \
    --env_cost:shield=0 \
    --env_reward_prestige_weight=0 \
    "$@"
