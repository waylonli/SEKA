import argparse, textwrap, torch

from transformers import AutoTokenizer

from src.model import SEKALLM
from utils import encode_with_markers          # same helper as before
import warnings
warnings.filterwarnings("ignore")

# ────────── CLI ───────────────────────────────────────────────────────
pa = argparse.ArgumentParser()
pa.add_argument('--model', default="pretrained/qwen2-1.5b-chat",
                help='HF id / local path')
pa.add_argument('--pos', default="projections/synthetic/qwen2-1.5b-chat_pos_proj.pt",
                help='positive (relevant) projector .pt')
pa.add_argument('--neg', default=None,
                help='optional negative (irrelevant) projector .pt')
pa.add_argument('--layers', default='last10',
                help="'all' / 'last4' / '0,4,19' …")
pa.add_argument('--prompt', required=True,
                help='plain‑text prompt (use *…* to highlight)')
pa.add_argument('--marker-start', default='*',
                help='highlight start marker (e.g. 👉 )')
pa.add_argument('--marker-end', default=None,
                help='highlight end marker; defaults to same as start')
pa.add_argument('--amplify-pos', default=1.5, type=float)
pa.add_argument('--amplify-neg', default=0.5, type=float)
pa.add_argument('--max-new', type=int, default=128)
pa.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'],
                default='auto')
pa.add_argument('--chat', action='store_true')
args = pa.parse_args()


if args.marker_end is None:
    args.marker_end = args.marker_start

tokenizer = AutoTokenizer.from_pretrained(args.model)

if args.chat:
    # apply chat-template to prompt
    chat_prompt = tokenizer.apply_chat_template(
        [{
            "role": "user",
            "content": args.prompt
        }],
        enable_thinking=True
    )

if "_tanh" in args.pos:
    feature_fn = "tanh"
elif "_elu" in args.pos:
    feature_fn = "elu"
elif "_squared" in args.pos:
    feature_fn = "squared-exponential"
else:
    feature_fn = None
# ────────────────── helper: encode with custom markers ─────────────────


# ────────────────── model wrapper ─────────────────────────────────────
ks = SEKALLM(
    args.model,
    pos_pt=args.pos,
    neg_pt=args.neg,
    layers=args.layers,
    amplify_pos=args.amplify_pos,
    amplify_neg=args.amplify_neg,
    feature_function=feature_fn,
    device=args.device
)

# print("\n=== PROMPT ===")
# print(textwrap.fill(args.prompt, 100))

# ids + mask with custom markers
ids, steering_mask, attention_mask = encode_with_markers(args.prompt, ks.tok,
                                args.marker_start, args.marker_end)
ids = ids.to(ks.device)

# ── baseline ──────────────────────────────────────────────────────────
baseline = ks.generate(ids, max_new_tokens=args.max_new, attention_mask=attention_mask)
print("\n--- baseline ---")
print(baseline)

# ── steering ──────────────────────────────────────────────────────────
print("\n--- steered ---")
steered = ks.generate(
    ids,
    steer=True,
    max_new_tokens=args.max_new,
    attention_mask=attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=False, temperature=0.0,
)
print(steered)

ks.remove_projection()          # tidy up
