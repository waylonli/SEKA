import os
import time
import torch
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model import SEKALLM, AdaptiveSEKALLM
from src.utils import encode_with_markers
from pastalib.pasta import PASTA, read_head_config
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Utility functions ---

def get_nvidia_smi_memory():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        mem_mb = int(output.decode("utf-8").strip().split("\n")[0])
        return mem_mb
    except Exception:
        return -1

def run_benchmark(name, get_model, tokenizer, prompts, highlighted_contexts, batch_size=10, max_new_tokens=60):
    mem_records = []
    speed_records = []
    answers = []
    nums_of_tokens = [len(tokenizer(p, return_tensors="pt").input_ids[0]) for p in prompts]
    model, mode = get_model()

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        hchunk = highlighted_contexts[start:start + batch_size]
        torch.cuda.reset_peak_memory_stats()
        smi_start = get_nvidia_smi_memory()
        batch_start = time.perf_counter()

        if mode == "original":
            enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(model.device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            decoded = tokenizer.batch_decode(
                out[:, enc.input_ids.shape[1]:], skip_special_tokens=True
            )
        elif mode == "pasta":
            inputs = tokenizer(chunk, return_tensors="pt", return_offsets_mapping=True, padding=True, truncation=True).to(DEVICE)
            offset_mapping = inputs.pop("offset_mapping")
            pasta = model
            with pasta.apply_steering(
                model=pasta.model,
                strings=chunk,
                substrings=hchunk,
                model_input=inputs,
                offsets_mapping=offset_mapping
            ) as steered_model:
                out = steered_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )
            decoded = tokenizer.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
        elif mode == "seka":
            ids, steer_mask, attention_mask = encode_with_markers(chunk, tokenizer)
            ids, steer_mask, attention_mask = ids.to(model.device), steer_mask.to(model.device), attention_mask.to(model.device)
            out = model.generate(
                ids=ids,
                steer_mask=steer_mask,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            # out is a list of decoded strings
            decoded = out if isinstance(out, list) else [out]
        elif mode == "adaptive_seka":
            ids, steer_mask, attention_mask = encode_with_markers(chunk, tokenizer)
            ids, steer_mask, attention_mask = ids.to(model.device), steer_mask.to(model.device), attention_mask.to(model.device)
            out = model.generate(
                ids=ids,
                steer_mask=steer_mask,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            decoded = out if isinstance(out, list) else [out]
        else:
            raise ValueError(f"Unknown model mode: {mode}")


        batch_time = time.perf_counter() - batch_start
        torch_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
        smi_mem = get_nvidia_smi_memory()

        if start > 0:
            mem_records.append({
                "torch_peak_MB": torch_mem,
                "smi_MB": smi_mem,
                "smi_MB_start": smi_start,
                "batch_size": len(chunk),
            })
            speed_records.append(batch_time)
            answers.extend(decoded)
            print(f"{name} | Batch {start//batch_size}: {batch_time:.2f}s, {torch_mem:.0f}MB torch, {smi_mem}MB smi")

    perf_df = pd.DataFrame(mem_records)
    perf_df["inference_time_sec"] = speed_records

    # calculate average for each batch
    print("Average inference time per sample:", sum(speed_records) / len(speed_records) / batch_size)
    print("Average torch peak memory usage:", perf_df['torch_peak_MB'].mean())
    print("Average smi memory usage:", perf_df['smi_MB'].mean())
    print("Average smi memory usage at start:", perf_df['smi_MB_start'].mean())
    print("Average number of tokens:", sum(nums_of_tokens) / len(nums_of_tokens))

    perf_df.to_csv(f"inference_profile_{name}.csv", index=False)
    return perf_df, answers

def plot_comparison(dfs, names):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for df, name in zip(dfs, names):
        axes[0].plot(df.index, df['torch_peak_MB'], label=f"{name} (torch mem)")
        axes[0].plot(df.index, df['smi_MB'], label=f"{name} (nvidia-smi mem)", linestyle='--', alpha=0.7)
        axes[1].plot(df.index, df['inference_time_sec'], label=name)
    axes[0].set_ylabel("Memory Usage (MB)")
    axes[1].set_ylabel("Inference Time (s)")
    axes[1].set_xlabel("Batch Number")
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("gpu_inference_profile_comparison.png")
    plt.show()

# --- Data: Replace with your 100-sample path ---
DATA_PATH = "data/lost_in_the_middle/30_total_documents/nq-open-30_total_documents_gold_at_0.jsonl"
with open(DATA_PATH) as f:
    examples = [json.loads(line) for line in f][:110]

def make_prompt_and_highlight(ex):
    # Your prompt logic: This example assumes "question", "ctxs", and highlights context 4:25
    def build_ctx(c, add_marker=True):
        ctx = "\n\n".join(f"{d['title']}\n{d['text']}" for d in c)
        if add_marker:
            # marker version: for SEKA and PASTA highlight
            ctx = "\n\n".join(f"{d['title']}\n{d['text']}" for d in c[:4]) + \
                  "\n\n**" + "\n\n".join(f"{d['title']}\n{d['text']}" for d in c[4:25]) + "**" + \
                  "\n\n" + "\n\n".join(f"{d['title']}\n{d['text']}" for d in c[25:])
        return ctx
    prompt = f"Directly answer in one short phrase without any other word.\n\nContext:\n{build_ctx(ex['ctxs'], False)}\n\nQuestion: {ex['question']}\n\nAnswer:"
    highlight = "\n\n".join(f"{d['title']}\n{d['text']}" for d in ex["ctxs"][4:25])
    return prompt, highlight

# Prepare
prompts, highlighted_contexts = zip(*(make_prompt_and_highlight(ex) for ex in examples))

# --- Model settings (update these) ---
MODEL_ID = "./pretrained/Qwen3-8B-Base"
# --- AdaSEKA ---
ADASEKA_EXPERT_CONFIG = "adaptive-seka-config/Qwen3-4B/Qwen3-4B-mindiff-0.4.json"
# --- SEKA ---
SEKA_POS = "projections/speed_memory/Qwen3-8B-Base_pos_proj.pt"
SEKA_NEG = "projections/speed_memory/Qwen3-8B-Base_neg_proj.pt"
SEKA_LAYERS = "all"
SEKA_AMPLIFY_POS = 0.1
SEKA_AMPLIFY_NEG = 0.0
# --- PASTA ---
PASTA_HEAD_CONFIG = "pastalib/config/kv_head/synthetic_diff0.1/Qwen3-8B-Base_head_config.json"
PASTA_ALPHA = 0.01
PASTA_SCALE_POSITION = "exclude"  # or "your_value"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")

# Original model function
def get_model_original():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE).eval()
    return model, "original"

# PASTA function
def get_model_pasta():
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",).to(DEVICE).eval()
    head_config = read_head_config(PASTA_HEAD_CONFIG)
    pasta = PASTA(
        base_model, tokenizer,
        head_config=head_config,
        alpha=PASTA_ALPHA,
        scale_position=PASTA_SCALE_POSITION,
    )
    return pasta, "pasta"

# SEKA function
def get_model_seka():
    seka_model = SEKALLM(
        MODEL_ID,
        pos_pt=SEKA_POS,
        neg_pt=SEKA_NEG,
        layers=SEKA_LAYERS,
        amplify_pos=SEKA_AMPLIFY_POS,
        amplify_neg=SEKA_AMPLIFY_NEG,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    return seka_model, "seka"

# Adaptive SEKA function
def get_model_adaptive_seka():
    adaptive_seka_model = AdaptiveSEKALLM(
        MODEL_ID,
        expert_paths=ADASEKA_EXPERT_CONFIG,
        layers=SEKA_LAYERS,
        top_k_singular=5,
        combination_method="weighted_top_k",
        amplify_factor=1.0,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    return adaptive_seka_model, "adaptive_seka"

# --- Run all three ---
print("\n--- ADAPTIVE SEKA ---")
df_adaptive_seka, _ = run_benchmark("adaptive_seka", get_model_adaptive_seka, tokenizer, prompts, highlighted_contexts)
print("\n--- SEKA ---")
df_seka, _ = run_benchmark("seka", get_model_seka, tokenizer, prompts, highlighted_contexts)
print("\n--- PASTA ---")
df_pasta, _ = run_benchmark("pasta", get_model_pasta, tokenizer, prompts, highlighted_contexts)
print("\n--- ORIGINAL MODEL ---")
df_orig, _ = run_benchmark("original", get_model_original, tokenizer, prompts, highlighted_contexts)

# --- Plot ---
plot_comparison([df_orig, df_pasta, df_seka], ["Original", "PASTA", "SEKA"])

print("✅ Finished benchmarking and plotting.")
