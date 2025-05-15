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
```
python benchmarks/eval_fact_gen.py \
    --model Qwen/Qwen3-8B \
    --data_path data/counterfact \
    --add_unmediated_fact True \
    --benchmarks efficacy paraphrase generation \
    --output_dir results/counterfact/Qwen3-8B \
    --max_new_tokens 128
```