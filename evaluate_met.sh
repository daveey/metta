./evaluate.sh \
    --env_width=30 \
    --env_height=30 \
    --env_num_agents=10 \
    --env_num_altars=5 \
    --env_num_chargers=10 \
    --env_num_generators=15 \
    --env_wall_density=0.05 \
    --env_reward_rank_steps=1000 \
    --env_reward:use=1 \
    --env_altar:reward=100 \
    --env_altar:cost=100 \
    --env_reward_prestige_weight=0 \
    "$@"
