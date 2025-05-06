import argparse, glob, json, logging, statistics, re, csv
import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, string, regex
from typing import List

# ────────── metric ───────────────────────────────────────────────────
def normalize_answer(s: str) -> str:
    def remove_articles(text):  return regex.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):  return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):            return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    norm_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) in norm_pred:
            return 1.0
    return 0.0

# ────────── prompt builders ──────────────────────────────────────────
def chat_prompt(ex, prefix):
    ctx = "\n\n".join(f"[{i}] {c['title']}\n{c['text']}" for i, c in enumerate(ex["ctxs"]))
    prefix_str = "**Pay more attention to the context in the middle.** " if prefix else ""
    user = (
        f"{prefix_str}Directly answer in one short phrase without any other word.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {ex['question']}"
    )
    return [{"role": "user", "content": user}]

def base_prompt(ex, prefix):
    ctx = "\n\n".join(f"[{i}] {c['title']}\n{c['text']}" for i, c in enumerate(ex["ctxs"]))
    prefix_str = "**Pay more attention to the context in the middle.** " if prefix else ""
    return (
        f"{prefix_str}Directly answer in one short phrase without any other word.\n\n"
        f"Question: {ex['question']}\n\nContext:\n{ctx}\n\nAnswer:"
    )

# ────────── main ─────────────────────────────────────────────────────
@torch.inference_mode()
def run(model_id, device, data_dir, chat,
        max_new_tokens, max_samples, batch_size, exp_name, prefix):
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModelForCausalLM.from_pretrained(model_id,
                                               torch_dtype=torch.bfloat16
                                              ).to(device).eval()

    base_dir   = Path("benchmarks/lost_in_the_middle")
    res_dir    = base_dir / "results" / exp_name
    pred_dir   = base_dir / "preds"   / exp_name
    res_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    csv_path = res_dir / f"lost_middle_{Path(model_id).name}.csv"
    with csv_path.open("w", newline="") as csv_f:
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(["gold_position", "exact_match"])

        # iterate over gold‑position files
        for fp in sorted(glob.glob(str(Path(data_dir) / "nq-open-*gold_at_*.jsonl"))):
            gold_pos = int(re.search(r"gold_at_(\d+)", fp).group(1))

            with open(fp) as fin:
                lines = fin.readlines()
            if max_samples > 0:
                lines = lines[:max_samples]
            examples = [json.loads(l) for l in lines]

            preds_file = pred_dir / f"preds_gold_at_{gold_pos}.jsonl"
            ems = []

            with preds_file.open("w") as pf:
                for start in tqdm(range(0, len(examples), batch_size),
                                  desc=f"gold@{gold_pos}"):
                    chunk = examples[start:start + batch_size]

                    prompts = ([tok.apply_chat_template(chat_prompt(e, prefix), tokenize=False)
                                for e in chunk] if chat
                               else [base_prompt(e, prefix) for e in chunk])

                    enc = tok(prompts, return_tensors="pt",
                              padding=True, truncation=True).to(device)

                    out = mod.generate(**enc,
                                       max_new_tokens=max_new_tokens,
                                       do_sample=False,
                                       pad_token_id=tok.eos_token_id)

                    prompt_lens = enc["attention_mask"].sum(1)

                    for j, ex in enumerate(chunk):
                        gen_ids = out[j, prompt_lens[j]:]
                        ans     = tok.decode(gen_ids,
                                             skip_special_tokens=True).strip()
                        ems.append(best_subspan_em(ans, ex["answers"]))
                        pf.write(json.dumps({"id": start + j,
                                             "answer": ans},
                                            ensure_ascii=False) + "\n")

            score = statistics.mean(ems)
            csv_writer.writerow([gold_pos, f"{score:.4f}"])
            logging.info("gold@%-2d  EM=%.3f", gold_pos, score)

    print(f"✅ curve → {csv_path}\n✅ preds → {pred_dir}/*.jsonl")


# ────────── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model", dest="model_id", required=True)
    p.add_argument("--chat", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--exp-name", default="default",
                   help="sub‑folder name for results/preds")
    p.add_argument("--prefix", action="store_true")
    args = p.parse_args()
    run(**vars(args))

