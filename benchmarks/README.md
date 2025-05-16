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
    --benchmarks efficacy paraphrase generation \
    --output_dir {path-to-output-dir} \
    --overwrite_output_dir
```