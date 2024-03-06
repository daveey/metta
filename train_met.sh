#/bin/bash -e

./train.sh \
    --env_width=25 \
    --env_height=25 \
    --env_num_agents=5 \
    --env_num_altars=1 \
    --env_num_chargers=3 \
    --env_num_generators=15 \
    --env_wall_density=0 \
    --env_reward_rank_steps=1000 \
    --env_reward:use=1 \
    --env_reward:metabolism=0 \
    --env_altar:reward=100 \
    --env_altar:cost=100 \
    --env_reward_prestige_weight=0 \
    --env_cost:attack=10000 \
    --env_cost:shield=10000 \
    --agent_fc_layers=4 \
    --agent_fc_size=512 \
    --rnn_num_layers=1 \
    --rnn_size=512 \
    --rnn_type=gru \
    "$@"
