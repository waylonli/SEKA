# utils.py  ────────────────────────────────────────────────────────────
import re, torch
from typing import List, Tuple, Union
from transformers import AutoTokenizer, AutoModel

# ╭──────────────────────── 1. encode_with_marker ───────────────────────╮
def encode_with_markers(text: Union[str, List[str]],
                        tokenizer: AutoTokenizer,
                        m_start: str = '**',
                        m_end: str = '**') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Accept either a single string or a list of strings and return
    `(ids[B, S], steer_mask[B, S], attention_mask[B, S])`
    padded to the longest sequence in the batch.
    """
    if isinstance(text, str):
        text = [text]  # unify to a batch

    ids_lst, steer_lst, attn_lst = [], [], []
    pattern = re.escape(m_start) + r'(.*?)' + re.escape(m_end)


    for sample in text:
        pieces, spans, last = [], [], 0
        for m in re.finditer(pattern, sample, flags=re.DOTALL):
            pieces.append(sample[last:m.start()])
            start_plain = sum(len(p) for p in pieces)
            span_content = m.group(1)
            pieces.append(span_content)
            spans.append((start_plain, start_plain + len(span_content)))
            last = m.end()

        pieces.append(sample[last:])
        plain_txt = ''.join(pieces)

        enc = tokenizer(
            plain_txt,
            return_offsets_mapping = True,
            add_special_tokens = False,
            padding = False,
        )

        ids_lst.append(enc['input_ids'])
        attn_lst.append(enc['attention_mask'])

        m = torch.zeros(len(enc['input_ids']), dtype=torch.bool)

        for s_char, e_char in spans:
            for tid, (cs, ce) in enumerate(enc['offset_mapping']):
                if cs >= s_char-1 and ce <= e_char:
                                    m[tid] = True

        steer_lst.append(m)

    # ---------- pad ----------
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    max_len = max(len(x) for x in ids_lst)

    def _left_pad(seq, pad_val):
        return [pad_val] * (max_len - len(seq)) + seq

    ids = torch.tensor([_left_pad(s, pad_id) for s in ids_lst])
    attn = torch.tensor([_left_pad(s, 0) for s in attn_lst])
    mask = torch.stack([
        torch.nn.functional.pad(m, (max_len - len(m), 0))  # left‑pad Boolean
        for m in steer_lst
    ])

    # import pdb; pdb.set_trace()
    return ids, mask, attn



# ╭──────────────────────── 2. attach_k_projection ────────────────────╮
def _parse_layers(spec: str, total: int):
    if spec == 'all':
        return list(range(total))
    if spec.startswith('last'):
        k = int(spec[4:]);   return list(range(total - k, total))
    if spec.startswith('first'):
        k = int(spec[5:]);   return list(range(k))
    print(f"[Key-Steering] using layers: {spec}")
    return [int(i) for i in spec.split(',')]


def _load_proj(path: str, device):
    obj = torch.load(path, map_location=device)

    # Unpack dict format
    if isinstance(obj, dict):
        layers = obj.get('layers', None)
        proj   = obj['proj'].to(device)
    else:
        layers = None
        proj   = obj.to(device)

    # Normalize to 4-D
    if proj.ndim == 2:
        # (d, d) → (1, 1, d, d)
        d = proj.size(0)
        proj = proj.unsqueeze(0).unsqueeze(0)
    elif proj.ndim == 3:
        # (L_sel, d, d) → (L_sel, 1, d, d)
        proj = proj.unsqueeze(1)
    elif proj.ndim == 4:
        # already (L_sel, H, d, d)
        pass
    else:
        raise ValueError(f"Unsupported proj dimension: {proj.ndim}")

    return layers, proj


def phi(x: torch.Tensor, name: str | None) -> torch.Tensor:
    if name is None:
        return x
    if name == 'squared-exponential':
        length_scale = 1
        return torch.exp(-1 * x ** 2 / (2 * length_scale ** 2))
    if name == 'tanh':
        return torch.tanh(x)
    if name == 'elu':
        return torch.where(x >= 0, x, torch.exp(x) - 1)
    raise ValueError(f'unknown feature_function {name}')


def phi_inv(x: torch.Tensor, name: str | None) -> torch.Tensor:
    if name is None:
        return x
    eps = torch.full_like(x, 1e-4)
    if name == "squared-exponential":
        length_scale = 1
        eps = torch.ones(x.shape) * 1e-4
        return -torch.log(torch.max(x, eps)) * 2 * length_scale ** 2
    if name == "tanh":
        x = torch.clamp(x, -1 + eps, 1 - eps)
        return torch.atanh(x)
    if name == "elu":
        pos = torch.clamp(x, min=0)
        neg = torch.clamp(x, max=0)
        return pos + torch.log(neg + 1)
    return x