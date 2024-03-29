#!/bin/bash -e

experiment=$1

for config in env_a5_25x25 env_a20_40x40 env_a100_100x100; do
    python -m framework.sample_factory.evaluate \
        --seed=0 \
        --config=evaluation \
        --config=$config \
        --save_video \
        --video_name="${config}.mp4" \
        --fps=8 \
        --experiment=$1 \
        "$@"
done

ln -s train_dir/$experiment/env_a20_40x40.mp4 ./train_dir/$experiment/replay.mp4
