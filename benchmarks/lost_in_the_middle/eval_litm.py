import argparse, glob, json, logging, statistics, re, csv
import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model import SEKALLM, AdaptiveSEKALLM
import torch, string, regex
from typing import List

from src.utils import encode_with_markers
from pastalib.pasta import PASTA, read_head_config
from anchoring import spa_tokenize, SPALogitsProcessor

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
def chat_prompt(ex, hlt_full, add_marker=True, anchor_mode=False):
    if add_marker:
        start_marker = "<anchor>" if anchor_mode else "**"
        end_marker = "</anchor>" if anchor_mode else "**"
        
        if not hlt_full:
            ctx = "\n\n".join(f"{c['title']}\n{c['text']}" for i, c in enumerate(ex["ctxs"][:4])) + \
                  "\n\n" + start_marker + "\n\n".join(f"{c['title']}\n{c['text']}" for i, c in enumerate(ex["ctxs"][4:25])) + end_marker + \
                  "\n\n" + "\n\n".join(f"{c['title']}\n{c['text']}" for i, c in enumerate(ex["ctxs"][25:]))
            highlighted_ctxs = "\n\n".join(f"{c['title']}\n{c['text']}" for i, c in enumerate(ex["ctxs"][4:25]))
        else:
            ctx = start_marker + "\n\n".join(f"{c['title']}\n{c['text']}" for i, c in enumerate(ex["ctxs"])) + end_marker
            highlighted_ctxs = "\n\n".join(f"{c['title']}\n{c['text']}" for i, c in enumerate(ex["ctxs"]))
    else:
        ctx = "\n\n".join(f"{c['title']}\n{c['text']}" for i, c in enumerate(ex["ctxs"]))
        highlighted_ctxs = ""
    user = (
        f"Directly answer in one short phrase without any other word.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {ex['question']}"
    )
    return [{"role": "user", "content": user}], highlighted_ctxs

def base_prompt(ex, hlt_full, add_marker=True, anchor_mode=False):
    if add_marker:
        start_marker = "<anchor>" if anchor_mode else "**"
        end_marker = "</anchor>" if anchor_mode else "**"
        
        if not hlt_full:
            ctx = "\n\n".join(f"{c['title']}\n{c['text']}" for i, c in enumerate(ex["ctxs"][:4])) + \
                  "\n\n" + start_marker + "\n\n".join(f"{c['title']}\n{c['text']}" for i, c in enumerate(ex["ctxs"][4:25])) + end_marker + \
                  "\n\n" + "\n\n".join(f"{c['title']}\n{c['text']}" for i, c in enumerate(ex["ctxs"][25:]))
            highlighted_ctxs = "\n\n".join(f"{c['title']}\n{c['text']}" for i, c in enumerate(ex["ctxs"][4:25]))
        else:
            ctx = start_marker + "\n\n".join(f"{c['title']}\n{c['text']}" for i, c in enumerate(ex["ctxs"])) + end_marker
            highlighted_ctxs = "\n\n".join(f"{c['title']}\n{c['text']}" for i, c in enumerate(ex["ctxs"]))
    else:
        ctx = "\n\n".join(f"{c['title']}\n{c['text']}" for i, c in enumerate(ex["ctxs"]))
        highlighted_ctxs = ""
    return (
        f"Directly answer in one short phrase without any other word.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {ex['question']}\n\nAnswer:"
    ), highlighted_ctxs

# ────────── main ─────────────────────────────────────────────────────
@torch.inference_mode()
def run(
        model_id,
        apply_seka,
        seka_pos,
        seka_neg,
        seka_amplify_pos,
        seka_amplify_neg,
        seka_layers,
        adaptive_seka,
        adaptive_expert_path,
        top_k_singular,
        combination_method,
        adaptive_amplify_factor,
        apply_pasta,
        head_config,
        pasta_alpha,
        scale_position,
        anchor,
        anchor_strength,
        device,
        data_dir,
        chat,
        max_new_tokens,
        max_samples,
        batch_size,
        exp_name,
        hlt_full
):
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    if adaptive_seka:
        adaptive_expert_path = json.load(open(adaptive_expert_path, "r"))
        mod = AdaptiveSEKALLM(
            model_id,
            expert_paths=adaptive_expert_path,
            layers=seka_layers,
            top_k_singular=top_k_singular,
            combination_method=combination_method,
            amplify_factor=adaptive_amplify_factor,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16
        )
    elif apply_seka:
        mod = SEKALLM(
            model_id,
            pos_pt=seka_pos,
            neg_pt=seka_neg,
            layers=seka_layers,
            amplify_pos=seka_amplify_pos,
            amplify_neg=seka_amplify_neg,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16
        )
    else:
        mod = AutoModelForCausalLM.from_pretrained(model_id,
                                                       torch_dtype=torch.bfloat16
                                                      ).to(device).eval()
        if apply_pasta:
            head_config = read_head_config(head_config)
            pasta = PASTA(
                mod,
                tok,
                head_config=head_config,
                alpha=pasta_alpha,
                scale_position=scale_position,
            )

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

                    prompts = ([tok.apply_chat_template(chat_prompt(e, hlt_full)[0], tokenize=False, enable_thinking=False)
                                for e in chunk] if chat
                               else [base_prompt(e, hlt_full)[0] for e in chunk])

                    highlighted_contexts = [tok.apply_chat_template(chat_prompt(e, hlt_full)[1], tokenize=False, enable_thinking=False)
                                            for e in chunk] if chat else \
                                            [base_prompt(e, hlt_full)[1] for e in chunk]


                    # enc = tok(prompts, return_tensors="pt",
                    #           padding=True, truncation=True).to(device)
                    ids, steering_mask, attention_mask = encode_with_markers(prompts, tok)
                    ids, steering_mask, attention_mask = ids.to(device), steering_mask.to(device), attention_mask.to(device)

                    if adaptive_seka:
                        out = mod.generate(
                            ids=ids,
                            steer_mask=steering_mask,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            temperature=0.0,
                            pad_token_id=tok.eos_token_id
                        )

                        for j, ex in enumerate(chunk):
                            ans = out[j]
                            ems.append(best_subspan_em(ans, ex["answers"]))
                            pf.write(json.dumps({"id": start + j,
                                                 "answer": ans,
                                                 "gold": ex["answers"]},
                                                ensure_ascii=False) + "\n")

                    elif apply_seka:
                        out = mod.generate(
                            ids=ids,
                            steer_mask=steering_mask,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            temperature=0.0,
                            pad_token_id=tok.eos_token_id
                        )

                        for j, ex in enumerate(chunk):
                            ans = out[j]
                            ems.append(best_subspan_em(ans, ex["answers"]))
                            pf.write(json.dumps({"id": start + j,
                                                 "answer": ans,
                                                 "gold": ex["answers"]},
                                                ensure_ascii=False) + "\n")

                    elif anchor:
                        # Anchor uses individual prompt processing with <anchor> markers
                        for j, ex in enumerate(chunk):
                            # Create anchor-specific prompt
                            anchor_prompt = ([tok.apply_chat_template(chat_prompt(ex, hlt_full, True, True)[0], tokenize=False, enable_thinking=False)
                                             for ex in [ex]] if chat
                                            else [base_prompt(ex, hlt_full, True, True)[0] for ex in [ex]])[0]
                            
                            # Create SPA logits processor
                            tok.pad_token_id = tok.eos_token_id
                            main_inputs, aux_inputs, mask_token = spa_tokenize(
                                prompt_with_anchors=anchor_prompt,
                                tokenizer=tok,
                                global_anchors=[],
                                device=device
                            )
                            anchor_processor = SPALogitsProcessor(
                                aux_model=mod,
                                aux_input_ids=aux_inputs,
                                strength=anchor_strength,
                                modulated_by_prob=False,
                                use_attention_mask=True,
                                mask_token=mask_token,
                                tokenizer=tok
                            )

                            # Generate text with SPA
                            out = mod.generate(
                                input_ids=main_inputs,
                                attention_mask=torch.ones_like(main_inputs),
                                logits_processor=[anchor_processor],
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                                pad_token_id=tok.eos_token_id,
                            )
                            
                            gen_ids = out[0, main_inputs.shape[-1]:]
                            ans = tok.decode(gen_ids, skip_special_tokens=True).strip()
                            ems.append(best_subspan_em(ans, ex["answers"]))
                            pf.write(json.dumps({"id": start + j,
                                                 "answer": ans,
                                                 "gold": ex["answers"]},
                                                ensure_ascii=False) + "\n")

                    elif not apply_pasta:
                        out = mod.generate(ids, attention_mask=attention_mask,
                                           max_new_tokens=max_new_tokens,
                                           do_sample=False,
                                           pad_token_id=tok.eos_token_id)

                        for j, ex in enumerate(chunk):
                            gen_ids = out[j, ids.shape[-1]:]
                            ans = tok.decode(gen_ids,
                                             skip_special_tokens=True).strip()
                            ems.append(best_subspan_em(ans, ex["answers"]))
                            pf.write(json.dumps({"id": start + j,
                                                 "answer": ans,
                                                 "gold": ex["answers"]},
                                                ensure_ascii=False) + "\n")
                    else:
                        prompts = ([tok.apply_chat_template(chat_prompt(e, hlt_full, False)[0], tokenize=False,
                                                            enable_thinking=False)
                                    for e in chunk] if chat
                                   else [base_prompt(e, hlt_full, False)[0] for e in chunk])
                        inputs = tok(
                            prompts, return_tensors="pt",
                            return_offsets_mapping=True,
                            truncation=True, padding=True
                        ).to(mod.device)
                        offset_mapping = inputs.pop("offset_mapping")
                        with pasta.apply_steering(
                                model=mod,
                                strings=prompts,
                                substrings=highlighted_contexts,
                                model_input=inputs,
                                offsets_mapping=offset_mapping
                        ) as steered_model:
                            out = steered_model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                pad_token_id=tok.eos_token_id,
                                do_sample=False,
                            )

                        for j, ex in enumerate(chunk):
                            gen_ids = out[j, inputs["input_ids"].shape[-1]:]
                            ans = tok.decode(gen_ids,
                                             skip_special_tokens=True).strip()
                            ems.append(best_subspan_em(ans, ex["answers"]))
                            pf.write(json.dumps({"id": start + j,
                                                 "answer": ans,
                                                 "gold": ex["answers"]},
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
    p.add_argument("--apply-seka", action="store_true")
    p.add_argument("--seka-pos", required=False)
    p.add_argument("--seka-neg", required=False)
    p.add_argument("--seka-layers", required=False, default="last10")
    p.add_argument("--seka-amplify-pos", required=False, type=float)
    p.add_argument("--seka-amplify-neg", required=False, type=float)
    p.add_argument("--adaptive-seka", action="store_true")
    p.add_argument("--adaptive-expert-path", required=False, help="Path to adaptive SEKA expert config JSON")
    p.add_argument("--top_k_singular", required=False, type=int, default=5, help="Top-k singular vectors")
    p.add_argument("--combination_method", required=False, default="weighted_top_k", help="Combination method")
    p.add_argument("--adaptive_amplify_factor", required=False, type=float, default=1.0, help="Amplify factor")
    p.add_argument("--apply-pasta", action="store_true")
    p.add_argument("--head_config", type=str, default=None, help="PASTA head config for steering")
    p.add_argument("--pasta_alpha", type=float, default=None, help="Scaling coefficient")
    p.add_argument("--scale_position", type=str, default=None, help="Steer the selected section or others")
    p.add_argument("--anchor", action="store_true", help="Enable anchor-based steering")
    p.add_argument("--anchor_strength", type=float, default=20.0, help="Anchor strength for SPA")
    p.add_argument("--chat", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--exp-name", default="default",
                   help="sub‑folder name for results/preds")
    p.add_argument("--hlt-full", action="store_true")
    args = p.parse_args()
    run(**vars(args))

