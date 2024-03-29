#/bin/bash -e


./train.sh \
    --env_width=25 \
    --env_height=25 \
    --env_num_agents=20 \
    --env_num_altars=1 \
    --env_num_chargers=10 \
    --env_num_generators=50 \
    --env_wall_density=0.01 \
    --env_reward_rank_steps=1000 \
    --env_altar:cost=100 \
    --env_reward_prestige_weight=0 \
    --env_cost:shield=0 \
    --agent_fc_layers=4 \
    --agent_fc_size=512 \
    --rnn_num_layers=1 \
    --rnn_size=512 \
    --rnn_type=gru \
    "$@"
