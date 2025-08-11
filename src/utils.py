# utils.py  ────────────────────────────────────────────────────────────
import re, torch
from typing import List, Tuple, Union
from transformers import AutoTokenizer, AutoModel

def encode_with_markers(
    text: Union[str, List[str]],
    tokenizer,
    m_start: str = '**',
    m_end: str = '**'
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Accepts either a single string or a list of strings and returns:
    (ids [B, S], steer_mask [B, S], attention_mask [B, S]),
    all correctly padded using the tokenizer.
    """
    if isinstance(text, str):
        text = [text]
    if not m_end:
        m_end = m_start

    pattern = re.escape(m_start) + r'(.*?)' + re.escape(m_end)
    stripped_texts = []
    spans_all = []

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
        stripped_texts.append(plain_txt)
        spans_all.append(spans)

    # Use tokenizer's own batching and special token logic
    enc = tokenizer(
        stripped_texts,
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    ids = enc['input_ids']
    attn = enc['attention_mask']

    mask = torch.zeros_like(ids, dtype=torch.bool)
    for b, spans in enumerate(spans_all):
        # Offsets are (token_start_char, token_end_char) for each token in sentence b
        for s_char, e_char in spans:
            for tid, (cs, ce) in enumerate(enc['offset_mapping'][b]):
                # Note: some tokenisers might have (0, 0) for special tokens or paddings
                if cs >= (s_char-1) and ce <= e_char and not (cs == ce == 0):
                    mask[b, tid] = True

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
    eps = torch.full_like(x, 1e-4).to(x.device)
    if name == "squared-exponential":
        length_scale = 1
        eps = torch.ones(x.shape).to(x.device) * 1e-4
        return -torch.log(torch.max(x, eps)) * 2 * length_scale ** 2
    if name == "tanh":
        x = torch.clamp(x, -1 + eps, 1 - eps)
        return torch.atanh(x)
    if name == "elu":
        pos = torch.clamp(x, min=0)
        neg = torch.clamp(x, max=0)
        return pos + torch.log(neg + 1)
    return x