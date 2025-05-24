## TODO
- [ ] Test biasbios
- [ ] Support chat templates
- [ ] Support more models

## CounterFact
### Setup
```
export PYTHONPATH=./
python -m spacy download en_core_web_sm
python -W ignore -m nltk.downloader punkt cmudict punkt_tab
python benchmarks/counterfact/download.py --path data/counterfact
```
### Run
Baseline
```
python benchmarks/eval_fact_gen.py \
    --model {model-name-or-path} \
    --data_path data/counterfact \
    --add_unmediated_fact True \
    --benchmarks efficacy paraphrase \
    --output_dir {path-to-output-dir} \
    --overwrite_output_dir
```
SEKA
```
python benchmarks/eval_fact_gen.py \
    --model {model-name-or-path} \
    --data_path data/counterfact \
    --add_unmediated_fact True \
    --benchmarks efficacy paraphrase \
    --output_dir {path-to-output-dir} \
    --seka \
    --pos {path-to-positive-projection} \
    --amplify-pos 2.0 \
    --layers all
```

## Biasbios
### Setup
```
export PYTHONPATH=./
python -m spacy download en_core_web_sm
python -W ignore -m nltk.downloader punkt cmudict punkt_tab
python benchmarks/biasbios/reformat_dataset.py \
    --biasbios_raw_path <path of BIOS.pkl> \
    --biasbios_save_file data/biasbios/biasbios.json 
```

### Run
Baseline
```
python benchmarks/eval_bias_gen.py \
    --model {model-name-or-path} \
    --data_path data/biosbias/biosbias.json \
    --output_dir {path-to-output-dir} \
    --batch_size 16 \
    --max_new_tokens 64
```
SEKA
```
python benchmarks/eval_bias_gen.py \
    --model {model-name-or-path} \
    --data_path data/biosbias/biosbias.json \
    --output_dir {path-to-output-dir} \
    --batch_size 16 \
    --max_new_tokens 64 \
    --seka \
    --pos {path-to-positive-projection} \
    --amplify_pos 1.0 \
    --layers all
```
