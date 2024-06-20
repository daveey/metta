#!/bin/bash -e

experiment=$1
shift

behavior_cfgs=$(find configs/env/mettagrid/behaviors/ -name "*.yaml" | sed 's|.*/behaviors/\(.*\)\.yaml|behaviors\1|')
train_cfgs="a5_25x25 a20_40x40 a20_r4_40x40 a20_b4_40x40"

for envcfg in $behavior_cfgs  $train_cfgs ; do
    echo "Generating video for $envcfg"
    video_name=${envcfg//\//_}
    python -m tools.evaluate \
        env=mettagrid/$envcfg \
        sample_factory=video \
        +sample_factory.video_name="${video_name}.mp4" \
        +sample_factory.experiment=$experiment \
        "$@"
done

cd train_dir/$experiment/ && rm -f replay.mp4 && ln -s a20_b4_40x40.mp4 replay.mp4
