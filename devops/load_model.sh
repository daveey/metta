#!/bin/bash -e

model=$1

if [ -z "$model" ]; then
    model="baseline"
fi

python -m sample_factory.huggingface.load_from_hub -r metta-ai/$model -d ./train_dir/

cd train_dir/$model
cp_path=$(ls checkpoint_p0/*.pth | tail -n 1)
echo "Linking $cp_path to train_dir/$model/latest.pth"
ln -s $cp_path latest.pth
