./evaluate.sh \
    --env_width=50 \
    --env_height=50 \
    --env_num_agents=50 \
    --env_num_altars=5:10 \
    --env_num_chargers=10 \
    --env_num_generators=5:20 \
    --env_wall_density=0.01 \
    --env_reward_rank_steps=1000 \
    --env_reward:use=1 \
    --env_altar:reward=100 \
    --env_altar:cost=100 \
    --env_cost:shield=0 \
    --env_reward_prestige_weight=0 \
    "$@"
