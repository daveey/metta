#/bin/bash -e

python -m train \
    --algo=APPO \
    --env=GDY-PowerGrid \
    --with_wandb=True \
    --num_workers=25 \
    --env_num_agents=20 \
    --env_width=20:100 \
    --env_height=20:100 \
    --agent_fc_layers=10 \
    --agent_fc_size=512 \
    --rnn_num_layers=2 \
    --rnn_size=256 \
    --rnn_type=gru \
    --rollout=256 \
    --batch_size=2048 \
    --decorrelate_experience_max_seconds=127  \
    --value_loss_coeff=0.9762413317396332 \
    --exploration_loss=symmetric_kl \
    --exploration_loss_coeff=0.002 \
    --policy_initialization=orthogonal \
    --initial_stddev=1 \
    --learning_rate=1.95-05 \
    --max_policy_lag=1713 \
    --nonlinearity=elu \
    --num_batches_per_epoch=7 \
    --num_batches_to_accumulate=1 \
    "$@"
