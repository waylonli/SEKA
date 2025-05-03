# test_inference_key.py
# -*- coding: utf-8 -*-
"""
Baseline vs. Spectral‑Editing Key Steering
────────────────────────────────────────────────────────────────────────────
Example:

python test_inference_key.py \
  --model qwen2-1.5b-chat \
  --rel   rel_proj.pt \
  --prompt "Write a script that prints hello. *Respond in JSON*."
"""
import argparse, textwrap, torch
from src.model.seka_llm import SEKALLM
from utils import encode_with_markers          # same helper as before
import warnings
warnings.filterwarnings("ignore")

# ────────── CLI ───────────────────────────────────────────────────────
pa = argparse.ArgumentParser()
pa.add_argument('--model',   default="pretrained/qwen2-1.5b-chat",
                help='HF id / local path')
pa.add_argument('--pos',     default="projections/synthetic/pos_proj.pt",
                help='positive (relevant) projector .pt')
pa.add_argument('--neg',   default=None,
                help='optional negative (irrelevant) projector .pt')
pa.add_argument('--layers',  default='last4',
                help="'all' / 'last4' / '0,4,19' …")
pa.add_argument('--prompt',  required=True,
                help='plain‑text prompt (use *…* to highlight)')
pa.add_argument('--marker-start', default='*',
                help='highlight start marker (e.g. 👉 )')
pa.add_argument('--marker-end',   default=None,
                help='highlight end marker; defaults to same as start')
pa.add_argument('--max-new', type=int, default=128)
pa.add_argument('--device',  choices=['auto', 'cuda', 'mps', 'cpu'],
                default='auto')
args = pa.parse_args()


if args.marker_end is None:
    args.marker_end = args.marker_start


# ────────────────── helper: encode with custom markers ─────────────────


# ────────────────── model wrapper ─────────────────────────────────────
ks = SEKALLM(args.model, device=args.device)

print("\n=== PROMPT ===")
print(textwrap.fill(args.prompt, 100))

# ids + mask with custom markers
ids, mask = encode_with_markers(args.prompt, ks.tok,
                                args.marker_start, args.marker_end)
ids = ids.to(ks.device)

# ── baseline ──────────────────────────────────────────────────────────
baseline = ks.generate(ids, max_new_tokens=args.max_new)
print("\n--- baseline ---")
print(baseline)

# ── steering ──────────────────────────────────────────────────────────
print("\n--- steered ---")
if args.neg:
    ks.attach_projection(rel_pt=args.pos,
                         irrel_pt=args.neg,
                         layers=args.layers,
                         mask_tensor=mask)
else:
    ks.attach_projection(rel_pt=args.pos,
                         layers=args.layers,
                         mask_tensor=mask)

print("\n")
steered = ks.generate(ids, max_new_tokens=args.max_new)
print(steered)

ks.remove_projection()          # tidy up
