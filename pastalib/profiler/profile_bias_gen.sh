#!/bin/bash

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=15

MODEL_NAME=Qwen3-4B-Base

start=$SECONDS

CONFIG_DIR=pastalib/config/train/${MODEL_NAME}
for entry in "$CONFIG_DIR"/*.json; do
    config_name="$(basename "$entry" .json)"

    if [ -f "pastalib/profiler/profile/$MODEL_NAME/biasbios/$config_name/result.json" ]; then
        echo "Done $config_name"
    else
        echo "==========================="
        echo "Processing: $config_name"
        echo "==========================="
        python benchmarks/eval_bias_gen.py \
            --model /mnt/data/models/$MODEL_NAME \
            --data_path data/biasbios/biasbios.json \
            --output_dir pastalib/profiler/profile/$MODEL_NAME/biasbios/$config_name \
            --batch_size 64 \
            --max_new_tokens 32 \
            --pasta \
            --head_config $entry \
            --pasta_alpha 0.01 \
            --scale_position exclude \
            --example_subset "4500:5000" \
            --overwrite_output_dir
    fi
done

duration=$(( SECONDS - start ))
echo "Script took $duration seconds"

