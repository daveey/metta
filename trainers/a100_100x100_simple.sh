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
    --env_reward:use=1 \
    --env_altar:reward=100 \
    --env_altar:cost=100 \
    --env_reward_prestige_weight=0 \
    --env_cost:shield=0 \
    --agent_fc_layers=4 \
    --agent_fc_size=512 \
    --rnn_num_layers=1 \
    --rnn_size=512 \
    --rnn_type=gru \
    "$@"
