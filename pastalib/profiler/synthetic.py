import json
import re
import string
from collections import Counter
from typing import Set, List, Tuple, Dict
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from pastalib.pasta import PASTA


def iter_examples(data_path, max_samples):
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            yield json.loads(line)


def get_triplets(ex: dict) -> list[tuple[str, str, str]]:
    return [
        (ex['context_1'], ex['context_2'], ex['question_1'], ex['answer_1']),
        (ex['context_2'], ex['context_1'], ex['question_2'], ex['answer_2'])
    ]


def normalize_answer(s: str) -> str:
    """
    Lower text, remove punctuation, articles and extra whitespace.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", ' ', text)
    def remove_punc(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    def white_space_fix(text):
        return ' '.join(text.split())

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_em(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def compute_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)


def evaluate_head(
    model,
    tokenizer,
    head_config,
    data_path: str,
    max_samples: int,
    max_new_tokens: int,
    alpha: float,
):
    pasta = PASTA(
        model=model,
        tokenizer=tokenizer,
        head_config=head_config, 
        alpha=alpha, # scaling coefficient
        scale_position="exclude", # downweighting unselected tokens
    )
    
    with open(data_path) as f:
        max_samples = min(max_samples, len(f.readlines()))

    total_em = 0
    total_f1 = 0.0
    n_examples = 0
    batch_size = 64

    batch_prompts: List[str] = []
    batch_rels:    List[str] = []
    batch_golds:   List[str] = []
        
    pbar = tqdm(total=max_samples, desc=f"Profiling {head_config}")
    for ex in iter_examples(data_path, max_samples):
        for rel_cxt, irr_cxt, question, gold in get_triplets(ex):
            prompt = f"Context: {irr_cxt} {rel_cxt}\nQuestion: {question}\nAnswer:"
            batch_prompts.append(prompt)
            batch_rels.append(rel_cxt)
            batch_golds.append(gold)

            if len(batch_prompts) >= batch_size:
                # prepare inputs
                inputs, offset_mapping = pasta.inputs_from_batch(batch_prompts)
                # steer & generate
                with pasta.apply_steering(
                    model=model,
                    strings=batch_prompts,
                    substrings=batch_rels,
                    model_input=inputs,
                    offsets_mapping=offset_mapping,
                ) as steered_model:
                    outputs = steered_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                # strip off the context length (padded to max_len)
                gen_ids = outputs[:, inputs.input_ids.size(1):] # shape (batch, new_tokens)
                preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

                # accumulate EM
                for pred, gold in zip(preds, batch_golds):
                    total_em += compute_em(pred, gold)
                    total_f1 += compute_f1(pred, gold)
                    n_examples += 1

                # clear for next batch
                batch_prompts.clear()
                batch_rels.clear()
                batch_golds.clear()
        pbar.update(1)
    pbar.close()

    # flush any remainder
    if batch_prompts:
        # prepare inputs
        inputs, offset_mapping = pasta.inputs_from_batch(batch_prompts)
        # steer & generate
        with pasta.apply_steering(
            model=model,
            strings=batch_prompts,
            substrings=batch_rels,
            model_input=inputs,
            offsets_mapping=offset_mapping,
        ) as steered_model:
            outputs = steered_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        # strip off the context length (padded to max_len)
        gen_ids = outputs[:, inputs.input_ids.size(1):] # shape (batch, new_tokens)
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # accumulate EM
        for pred, gold in zip(preds, batch_golds):
            total_em += compute_em(pred, gold)
            total_f1 += compute_f1(pred, gold)
            n_examples += 1

        # clear for next batch
        batch_prompts.clear()
        batch_rels.clear()
        batch_golds.clear()

    # Report
    em_score = total_em / n_examples * 100
    f1_score = total_f1 / n_examples * 100
    print(f"Evaluated {n_examples} examples")
    print(f"Exact Match (EM): {em_score:.2f}%")
    print(f"F1 Score: {f1_score:.2f}%")
    return em_score, f1_score

def profile_heads(
    model,
    tokenizer,
    task_data_paths: List[str],
    k: int,
    max_samples_per_task: int = 1000,
    max_new_tokens: int = 32,
    alpha: float = 0.01,
) -> Set[Tuple[int,int]]:
    """
    For each task, find top-k heads by EM, then intersect them.
    Returns the set H of (layer, head) pairs.
    """
    # figure out how many layers/heads
    num_layers = len(model.model.layers)
    num_heads  = model.config.num_attention_heads

    # all single-head configurations
    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]

    # store the top-k for each task
    topk_per_task: List[Set[Tuple[int,int]]] = []

    for path in task_data_paths:
        print(f"\nProfiling task {path!r}...")
        scores: Dict[Tuple[int,int], float] = {}
        for (l, h) in tqdm(all_heads):
            em = evaluate_head(
                model, tokenizer,
                head_config={l: [h]},
                data_path=path,
                max_samples=max_samples_per_task,
                max_new_tokens=max_new_tokens,
                alpha=alpha,
            )
            scores[(l, h)] = em
        # sort by descending EM
        sorted_heads = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        topk = { head for head, _ in sorted_heads[:k] }
        print(f"  -> top-{k} heads for this task:", topk)
        topk_per_task.append(topk)

    # intersect across all tasks
    H = set.intersection(*topk_per_task)
    print(f"\nFinal selected head set H (intersection): {H}")
    
    # Convert to { layer_str: [head, head, …], … }
    head_config: Dict[str, List[int]] = {}
    for layer, head in H:
        layer_key = str(layer)
        head_config.setdefault(layer_key, []).append(head)
    for layer_key in head_config:
        head_config[layer_key].sort()
    
    print("Final head_config =", head_config)
    return head_config


def main():
     # 1) load model + tokenizer
    MODEL_NAME = "/mnt/data/models/Qwen3-4B-Base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager"
    )

    # 2) point at your m tasks (e.g. m different JSONL files)
    tasks = [
        "/mnt/data/nyc/SEKA/data/synthetic/pair_qa_new.jsonl",
    ]

    # 3) pick how many heads to keep per task, e.g. k=10
    k = 500

    # 4) run profiler
    head_config = profile_heads(
        model,
        tokenizer,
        task_data_paths=tasks,
        k=k,
        max_samples_per_task=1000,
        max_new_tokens=16,
        alpha=0.01,
    )
    
    # 5) save
    save_path = "/mnt/data/nyc/SEKA/pastalib/config/Qwen3-4B-Base.json"
    with open(save_path, 'w') as f:
        json.dump(head_config, f, indent=4)

if __name__ == "__main__":
    main()