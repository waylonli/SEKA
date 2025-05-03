"""
python build_key_projection.py \
   --model pretrained/qwen2-1.5b-chat \
   --layers "all" \
   --json data/pair_qa.json \
   --top-pct 0.97
"""
import argparse, json, random, tqdm, torch
from   transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)

# ────────── CLI ───────────────────────────────────────────────────────
pa = argparse.ArgumentParser()
pa.add_argument('--model',      required=True)
pa.add_argument('--layers',     default='all',   help="comma | 'last4' | 'all'")
pa.add_argument('--json',       required=True)
pa.add_argument('--samples',    type=int,  default=100)
pa.add_argument('--top-pct',    type=float,default=0.97)
args = pa.parse_args()

# ────────── layer selection ───────────────────────────────────────────
def parse_layers(spec, n_total):
    if spec == 'all':               return list(range(n_total))
    if spec.startswith('last'):     k = int(spec[4:]);  return list(range(n_total-k, n_total))
    return [int(x) for x in spec.split(',')]

# ────────── model ─────────────────────────────────────────────────────
tok   = AutoTokenizer.from_pretrained(
    args.model,
    max_length=9000
)
model = AutoModelForCausalLM.from_pretrained(args.model).half().cuda().eval()

N_LAYERS = len(model.model.layers)
LAYERS   = parse_layers(args.layers, N_LAYERS)
H        = model.config.num_attention_heads
KvH      = model.config.num_key_value_heads
head_dim = model.config.hidden_size // H
d_kv     = head_dim * KvH                              # 256 for Qwen‑1.5 B

print(f"› collecting keys for layers {LAYERS}")

# ────────── helpers ───────────────────────────────────────────────────
def span_ids(text, sub):
    if sub in text:
        a, b = text.index(sub), text.index(sub) + len(sub)
        offs = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        return [i for i, (s, e) in enumerate(offs) if s >= a and e <= b]
    return None

@torch.no_grad()
def layer_keys(text, idx):
    ids  = tok(text, return_tensors='pt', add_special_tokens=False).to(model.device)
    hidd = model(**ids, use_cache=False, output_hidden_states=True).hidden_states
    out  = []
    for ℓ in LAYERS:
        h = hidd[ℓ][0]                                       # (seq,d_model)  ── input to layer ℓ
        k = model.model.layers[ℓ].self_attn.k_proj(h)        # (seq,d_kv)
        out.append(k[idx].float())                           # (n_sel,d_kv)
    return out

# ────────── collect keys ──────────────────────────────────────────────
rel_buf, irrel_buf = [ [] for _ in LAYERS ], [ [] for _ in LAYERS ]
data = json.load(open(args.json));  random.shuffle(data)

for ex in tqdm.tqdm(data[:args.samples], desc="collect"):
    rel_ctx   = ex["relevant_context"]
    rel_tok   = ex["answer"]
    irrel_ctx = ex["irrelevant_context"]
    q         = ex["question"]

    ctx = f"{rel_ctx}\n{irrel_ctx}\nQuestion: {q}" if random.random() < .5 \
        else f"{irrel_ctx}\n{rel_ctx}\nQuestion: {q}"

    idx_pos = span_ids(ctx, rel_tok)
    idx_neg = span_ids(ctx, irrel_ctx)

    if idx_pos:
        for buf, k in zip(rel_buf, layer_keys(ctx, idx_pos)):
            buf.append(k)
    if idx_neg:
        for buf, k in zip(irrel_buf, layer_keys(ctx, idx_neg)):
            buf.append(k)

# ────────── PCA → projectors ─────────────────────────────────────────
def build_proj(mat: torch.Tensor, pct: float, max_retry: int = 5):
    mat = (mat - mat.mean(0, keepdim=True)).double().cpu()
    d   = mat.size(1);  eps = 1e-8

    for _ in range(max_retry):
        try:
            mat_f32 = mat.float()
            C = (mat_f32.T @ mat_f32) / mat_f32.size(0)
            U, S, _ = torch.linalg.svd(C, full_matrices=False)  # float32 SVD
            break
        except RuntimeError: eps *= 10
    else:
        S, U = torch.linalg.eigh((mat.T @ mat) / mat.size(0) + eps*torch.eye(d))
        S, U = torch.flip(S, [0]), torch.flip(U, [1])

    k = (torch.cumsum(S, 0) / S.sum() < pct).sum().item() + 1
    return (U[:, :k] @ U[:, :k].T).to(torch.float16)           # (d,d)

rel_proj_stack, irrel_proj_stack = [], []
for ℓ, (pos, neg) in enumerate(zip(rel_buf, irrel_buf)):
    if not pos:   raise RuntimeError(f"no positive data for layer {LAYERS[ℓ]}")
    if not neg:   raise RuntimeError(f"no negative data for layer {LAYERS[ℓ]}")
    rel_proj_stack.append(build_proj(torch.cat(pos, 0), args.top_pct))
    irrel_proj_stack.append(build_proj(torch.cat(neg, 0), args.top_pct))

rel_proj   = torch.stack(rel_proj_stack)    # (L,d,d)
irrel_proj = torch.stack(irrel_proj_stack)

torch.save({'layers': LAYERS, 'proj': rel_proj.cpu()},   f'projections/synthetic/{args.model.split('/')[-1]}_pos_proj.pt')
torch.save({'layers': LAYERS, 'proj': irrel_proj.cpu()}, f'projections/synthetic/{args.model.split('/')[-1]}_neg_proj.pt')

print("✓ saved pos_proj.pt",   tuple(rel_proj.shape))
print("✓ saved neg_proj.pt", tuple(irrel_proj.shape))
