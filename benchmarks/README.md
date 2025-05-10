## TODO
- [ ] Test biasbios
- [ ] Support chat templates
- [ ] Support more models

## CounterFact
### Setup
```
python -m spacy download en_core_web_sm
python -W ignore -m nltk.downloader punkt cmudict punkt_tab
python benchmarks/counterfact/download.py --path /home/ycniu/SEKA/data/counterfact
```
### Run
```
export PYTHONPATH=./
python benchmarks/eval_fact_gen.py \
    --model Qwen/Qwen3-1.7B \
    --data_path /home/ycniu/SEKA/data/counterfact \
    --add_unmediated_fact True \
    --benchmarks efficacy paraphrase generation \
    --output_dir /home/ycniu/SEKA/results/counterfact/Qwen3-1.7B \
    --max_new_tokens 128
```