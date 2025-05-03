# utils.py  ────────────────────────────────────────────────────────────
import re, torch
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel

# ╭──────────────────────── 1. encode_with_marker ───────────────────────╮
def encode_with_markers(text: str,
                        tokenizer: AutoTokenizer,
                        m_start: str,
                        m_end: str):
    """
    Return (input_ids[1,seq], mask[seq]) where mask marks tokens that were
    inside <m_start> ... <m_end> spans.  Markers themselves are stripped.
    Works even when start==end (e.g. '*').
    """
    # 1) locate marker spans & build plain text
    pattern = re.escape(m_start) + r'(.*?)' + re.escape(m_end)
    pieces, spans, last = [], [], 0
    for m in re.finditer(pattern, text, flags=re.DOTALL):
        pieces.append(text[last:m.start()])          # plain before span
        start_plain = sum(len(p) for p in pieces)
        span_content = m.group(1)
        pieces.append(span_content)
        spans.append((start_plain, start_plain + len(span_content)))
        last = m.end()
    pieces.append(text[last:])
    plain_txt = ''.join(pieces)

    # 2) tokenise plain text and build mask
    enc = tokenizer(plain_txt,
                    return_offsets_mapping=True,
                    add_special_tokens=False)
    ids  = torch.tensor([enc['input_ids']])
    mask = torch.zeros(len(enc['input_ids']), dtype=torch.bool)

    for s_char, e_char in spans:
        for tid, (cs, ce) in enumerate(enc['offset_mapping']):
            if cs >= s_char and ce <= e_char:
                mask[tid] = True

    return ids, mask



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
    """
    允许两种格式：
      • torch tensor  :  shape (d,d)  或  (L_sel,d,d)
      • dict{layers,proj}
          layers : List[int] 与 proj 首维对应
          proj   : Tensor (L_sel,d,d)
    """
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict):
        return obj['layers'], obj['proj'].to(device)
    if obj.ndim == 2:          # 通用到所有层
        return None, obj[None].to(device)         # (1,d,d)
    if obj.ndim == 3:
        return list(range(obj.size(0))), obj.to(device)
    raise ValueError("Unsupported projection format")