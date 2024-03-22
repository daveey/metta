#!/bin/bash -e

model=$1

if [ -z "$model" ]; then
    model="baseline"
fi

python -m sample_factory.huggingface.load_from_hub -r metta-ai/$model -d ./train_dir/
