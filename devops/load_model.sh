#!/bin/bash -e

model=$1

if [ -z "$model" ]; then
    model="baseline"
fi

python -m sample_factory.huggingface.load_from_hub -r metta-ai/$model -d ./train_dir/

cp_path=$(ls train_dir/$model/checkpoint_p0/*.pth | tail -n 1)
echo "Linking $cp_path to train_dir/$model/latest.pth"
ln -s $cp_path train_dir/$model/latest.pth
