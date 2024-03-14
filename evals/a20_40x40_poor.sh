./evaluate.sh \
    --env_width=40 \
    --env_height=40 \
    --env_num_agents=20 \
    --env_num_altars=2 \
    --env_num_chargers=5 \
    --env_num_generators=5 \
    --env_reward_rank_steps=1000 \
    --env_altar:cost=100 \
    --env_cost:shield=0 \
    --env_reward_prestige_weight=0 \
    "$@"
