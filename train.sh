#/bin/bash -e

pkill -9 wandb
pkill -9 python
git pull
python -m train \
    --algo=APPO \
    --env=GDY-PowerGrid \
    --with_wandb=True \
    --wandb_user=platypus \
    --wandb_project=sample_factory \
    --env_num_agents=5 \
    --env_width=20:40 \
    --env_height=20:40 \
    --env_num_altars=1:5 \
    --env_num_chargers=1:5 \
    --env_num_generators=5:20 \
    --env_wall_density=0:0.15 \
    --env_reward_rank_steps=10 \
    --agent_fc_layers=3 \
    --agent_fc_size=512 \
    --agent_embedding_size=64 \
    --agent_embedding_layers=3 \
    --agent_attention_size=512 \
    --agent_attention_layers=2 \
    --rnn_num_layers=1 \
    --rnn_size=512 \
    --rnn_type=gru \
    --recurrence=32 \
    --rollout=256 \
    --batch_size=16384 \
    --decorrelate_experience_max_seconds=150  \
    --value_loss_coeff=0.976 \
    --exploration_loss=symmetric_kl \
    --exploration_loss_coeff=0.002 \
    --policy_initialization=orthogonal \
    --learning_rate=0.0001 \
    --max_policy_lag=2000 \
    --nonlinearity=elu \
    --normalize_input=False \
    "$@"
