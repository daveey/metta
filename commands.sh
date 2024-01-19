# copy models from s3
aws s3 sync s3://metta-ai/vast/workspace/metta/train_dir ./train_dir

# train
python -m envs.griddly.gridman.train \
    --algo=APPO \
    --env=GDY-GridmanMultiAgent \
    --experiment=gm.rnn=2048 \
    --rnn_size=2048 \
    --with_wandb=True \
    --num_workers=25 \

