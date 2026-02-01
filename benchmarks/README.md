# Benchmarks

This folder hosts the evaluation harness used to produce the **BiasBios**, **CounterFact**, and **PronChange** results reported in *SEKA: Spectral Editing Key Amplification* (ICLR 2026). Follow the steps below to recreate the paper numbers end-to-end.

## 1. Prepare the datasets

Run the preprocessing scripts once to materialise the JSON files that the evaluation drivers expect:

```bash
# BiasBios professions
python benchmarks/biasbios/reformat_dataset.py \
  --biasbios_raw_path <path/to/BIOS.pkl> \
  --biasbios_save_file data/biasbios/biasbios.json

# CounterFact factual editing
python benchmarks/counterfact/download.py --path data/counterfact
```

PronChange reuses the processed BiasBios JSON, so no additional data preparation is required.

## 2. Obtain SEKA projection banks

You can either regenerate the projection banks or download the pre-built archives released with the paper:

- **Generate locally** using the builders in `src/custom_builders/`. For example:
  ```bash
  python src/custom_builders/synthetic_qa_builder.py \
    --model pretrained/Qwen3-4B-Base \
    --data data/synthetic/pair_qa_new.jsonl \
    --output_dir projections/biasbios/Qwen3-4B-Base \
    --max_samples 200 \
    --min_diff 0.20 \
    --top_pct 0.90
  ```
- **Download** the projection packs shipped with the camera-ready (links forthcoming) and unpack them under `projections/<task>/<model>/`.

Each benchmark expects `*_pos_proj.pt` (and optionally `*_neg_proj.pt`) to be available at the paths you pass via `--pos` / `--neg`.

## 3. Run the evaluation drivers

Use the Python scripts directly to reproduce the main results. Swap in the model/projection pair of interest; the commands below match the settings used in the paper.

### BiasBios

```bash
python benchmarks/eval_bias_gen.py \
  --model pretrained/Qwen3-4B-Base \
  --data_path data/biasbios/biasbios.json \
  --output_dir benchmarks/biasbios/results/seka-qwen3-4b \
  --overwrite_output_dir \
  --batch_size 32 \
  --max_new_tokens 64 \
  --seka \
  --pos projections/biasbios/Qwen3-4B-Base_pos_proj.pt \
  --neg projections/biasbios/Qwen3-4B-Base_neg_proj.pt \
  --amplify_pos 1.0 \
  --amplify_neg 0.8 \
  --layers last10
```

### CounterFact

```bash
python benchmarks/eval_fact_gen.py \
  --model pretrained/Qwen3-4B-Base \
  --data_path data/counterfact \
  --output_dir benchmarks/counterfact/results/seka-qwen3-4b \
  --overwrite_output_dir \
  --benchmarks efficacy paraphrase \
  --add_unmediated_fact True \
  --batch_size 32 \
  --max_new_tokens 64 \
  --seka \
  --pos projections/counterfact/Qwen3-4B-Base_pos_proj.pt \
  --neg projections/counterfact/Qwen3-4B-Base_neg_proj.pt \
  --amplify_pos 1.56 \
  --amplify_neg 0.0 \
  --layers last10
```

### PronChange

```bash
python benchmarks/eval_biasbios_instruction.py \
  --model pretrained/gemma-3-4b-pt \
  --data_path data/biasbios/biasbios.json \
  --task pronchange \
  --output_dir benchmarks/pronchange/results/seka-gemma-3-4b \
  --overwrite_output_dir \
  --batch_size 32 \
  --max_new_tokens 256 \
  --seka \
  --pos projections/pronchange/gemma-3-4b-pt_pos_proj.pt \
  --neg projections/pronchange/gemma-3-4b-pt_neg_proj.pt \
  --amplify_pos 0.40 \
  --amplify_neg 0.00 \
  --layers last10 \
  --example_subset 28297:29297  # optional slice used in the paper
```

Each run emits `metric_result.json` (headline metrics) and task-specific logs inside the specified `--output_dir`.

## 4. Key arguments

| Argument | Description |
| --- | --- |
| `--model` | HF identifier or local path under `pretrained/` |
| `--data_path` | Path to the processed dataset JSON / folder |
| `--pos`, `--neg` | Paths to SEKA projection tensors (`*_pos_proj.pt`, `*_neg_proj.pt`) |
| `--amplify_pos`, `--amplify_neg` | Steering coefficients (values from Tables 2–4 of the paper) |
| `--layers` | Layer subset (`last10`, `all`, `0,1,2`, etc.) parsed by `_parse_layers` |
| `--example_subset` | Optional `start:end` slice for quick sanity checks |

## 5. Troubleshooting

- **Projection mismatch** – Ensure you load projections trained for the *same model family* and *task* you are evaluating.
- **Negative amplitude behaviour** – If you do not wish to use a negative projection, omit the `--neg` flag entirely; passing the flag with `--amplify_neg 0.0` will attenuate the positive term.
- **Cluster modules** – The example scripts assume CUDA modules named `cuda/12.x`; adapt to your HPC environment if necessary.

That’s it! The commands above recreate the CounterFact, BiasBios, and PronChange numbers reported in *SEKA: Spectral Editing Key Amplification*.
