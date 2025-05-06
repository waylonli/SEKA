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
def chat_prompt(ex):
    ctx = "\n\n".join(f"[{i}] {c['title']}\n{c['text']}" for i, c in enumerate(ex["ctxs"]))
    user = (
        "Directly answer in one short phrase without any other word.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {ex['question']}"
    )
    return [{"role": "user", "content": user}]

def base_prompt(ex):
    ctx = "\n\n".join(f"[{i}] {c['title']}\n{c['text']}" for i, c in enumerate(ex["ctxs"]))
    return (
        "Directly answer in one short phrase without any other word.\n\n"
        f"Question: {ex['question']}\n\nContext:\n{ctx}\n\nAnswer:"
    )

# ────────── main ─────────────────────────────────────────────────────
@torch.inference_mode()
def run(model_id, device, data_dir, chat, max_new_tokens, max_samples, batch_size):
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to(device).eval()

    csv_path = Path("benchmarks/lost_in_the_middle/results") / f"lost_middle_{Path(model_id).name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["gold_position", "exact_match"])

        # iterate over each gold‑position file
        for fp in sorted(glob.glob(str(Path(data_dir) / "nq-open-*gold_at_*.jsonl"))):
            gold_pos = int(re.search(r"gold_at_(\d+)", fp).group(1))
            ems, prompts, answers = [], [], []

            # ─ collect examples ───────────────────────────────────────
            with open(fp) as fin:
                all_lines = fin.readlines()[: max_samples] if max_samples > 0 else fin.readlines()
                examples  = [json.loads(l) for l in all_lines]

            # ─ batch inference ────────────────────────────────────────
            for i in tqdm(range(0, len(examples), batch_size)):
                chunk = examples[i : i + batch_size]

                # prompts
                if chat:
                    prompts = [
                        tok.apply_chat_template(chat_prompt(ex), tokenize=False) for ex in chunk
                    ]
                else:
                    prompts = [base_prompt(ex) for ex in chunk]

                enc = tok(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)

                # ---- run batched generation ----
                out = mod.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )

                # length of each prompt before generation (no‑pad tokens)
                prompt_lens = enc["attention_mask"].sum(dim=1)  # (batch,)

                gen_texts = []
                for j in range(out.size(0)):
                    start = prompt_lens[j].item()  # int
                    gen_ids = out[j, start:]  # generated part
                    gen_texts.append(tok.decode(gen_ids,
                                                skip_special_tokens=True).strip())

                answers.extend(gen_texts)

            # ─ scoring ────────────────────────────────────────────────
            for ex, pred in zip(examples, answers):
                ems.append(best_subspan_em(pred, ex["answers"]))

            score = statistics.mean(ems) if ems else 0.0
            logging.info("gold@%-2d  EM=%.3f", gold_pos, score)
            writer.writerow([gold_pos, f"{score:.4f}"])

    print(f"✅  results saved to {csv_path}")


# ────────── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model", dest="model_id", required=True)
    p.add_argument("--chat", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--max-samples", type=int, default=-1,
                   help="‑1 → use every example in the file")
    p.add_argument("--batch-size", type=int, default=32,
                   help="examples per forward pass")
    args = p.parse_args()
    run(**vars(args))
