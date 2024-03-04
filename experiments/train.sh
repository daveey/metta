#/bin/bash -e

pkill -9 wandb
pkill -9 python
git pull

python -m train \
    --algo=APPO \
    --env=GDY-PowerGrid \
    --with_wandb=True \
    --num_workers=25 \
    --agent_fc_layers=4 \
    --agent_fc_size=512 \
    --rnn_num_layers=1 \
    --rnn_size=512 \
    --rnn_type=gru \
    --rollout=256 \
    --batch_size=16384 \
    --decorrelate_experience_max_seconds=150  \
    --value_loss_coeff=0.976 \
    --exploration_loss=symmetric_kl \
    --exploration_loss_coeff=0.002 \
    --policy_initialization=orthogonal \
    --learning_rate=0.0000195 \
    --max_policy_lag=2000 \
    --nonlinearity=elu \
    "$@"
