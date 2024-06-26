#!/bin/bash -e

experiment=$1
shift

for envcfg in a5_25x25 a20_40x40 a20_r4_40x40 a20_b4_40x40; do
    echo "Generating video for $envcfg"
    python -m tools.evaluate \
        env=mettagrid/$envcfg \
        sample_factory=video \
        +sample_factory.video_name="${envcfg}.mp4" \
        +sample_factory.experiment=$experiment \
        "$@"
done

cd train_dir/$experiment/ && rm -f replay.mp4 && ln -s a20_b4_40x40.mp4 replay.mp4
