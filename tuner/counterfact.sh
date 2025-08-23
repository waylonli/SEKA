#!/bin/bash

export CUDA_VISIBLE_DEVICES=15
export PYTHONPATH=./

# python tuner/main.py \
#     --build_method synth \
#     --build_data_path data/synthetic/pair_qa_new.jsonl \
#     --model_type varnorm_seka \
#     --model_path /mnt/data/models/Qwen3-4B-Base \
#     --task counterfact \
#     --eval_data_path data/counterfact/counterfact.jsonl \
#     --eval_example_subset 4500:5000 \
#     --batch_size 16 \
#     --top_pct_range 0.7 1.0 \
#     --min_diff_range 0.001 0.3 \
#     --n_trials 100 2>&1 | tee tuner/varnorm_counterfact.log

python tuner/main.py \
    --build_method synth \
    --build_data_path data/synthetic/pair_qa_new.jsonl \
    --model_type seka \
    --model_path /mnt/data/models/Qwen3-4B-Base \
    --task counterfact \
    --eval_data_path data/counterfact/counterfact.jsonl \
    --eval_example_subset 4500:5000 \
    --batch_size 16 \
    --top_pct_range 0.8 1.0 \
    --min_diff_range 0.1 1.0 \
    --use_proj_neg \
    --amp_pos_range 1.5 3.0 \
    --amp_neg_range 0.0 1.0 \
    --n_trials 100 2>&1 | tee tuner/seka_counterfact.log
    
