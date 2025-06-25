#!/bin/bash

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=14

MODEL_NAME=Qwen3-14B-Base

start=$SECONDS

CONFIG_DIR=pastalib/config/train/${MODEL_NAME}
for entry in "$CONFIG_DIR"/*.json; do
    config_name="$(basename "$entry" .json)"

    if [ -d "pastalib/profiler/profile/$MODEL_NAME/counterfact/$config_name" ]; then
        echo "Done $config_name"
    else
        echo "==========================="
        echo "Processing: $config_name"
        echo "==========================="
        python benchmarks/eval_fact_gen.py \
            --model /mnt/data/models/$MODEL_NAME \
            --data_path data/counterfact \
            --add_unmediated_fact True \
            --benchmarks efficacy paraphrase \
            --output_dir pastalib/profiler/profile/$MODEL_NAME/counterfact/$config_name \
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
