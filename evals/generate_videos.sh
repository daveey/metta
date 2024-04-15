#!/bin/bash -e

experiment=$1
shift

for config in env_a5_25x25 env_a20_40x40 env_a100_100x100; do
    python -m tool \
        --seed=0 \
        --config=evaluation \
        --config=$config \
        --save_video \
        --video_name="${config}.mp4" \
        --fps=8 \
        --experiment=$experiment \
        "$@"
done

cd train_dir/$experiment/ && ln -s env_a20_40x40.mp4 replay.mp4
