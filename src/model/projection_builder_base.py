# build_key_projection.py
# -*- coding: utf-8 -*-
"""
python build_key_projection.py \
   --model pretrained/qwen2-1.5b-chat \
   --layers all \
   --json   data/pair_qa.json \
   --samples 100 \
   --top-pct 0.97
"""
from __future__ import annotations
import argparse, json, random, tqdm, torch, abc, pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
torch.set_grad_enabled(False)


# ────────────────────── BASE CLASS ────────────────────────────────────
class ProjectionBuilderBase(abc.ABC):
    """
    Dataset‑agnostic scaffolding that
      • loads the HF model
      • builds per‑layer positive / negative projectors
    Sub‑classes implement `iter_examples()` that yields triples:
        ctx_str, positive_span_str, negative_span_str
    """

    def __init__(self,
                 model_path: str,
                 layers: str,
                 *,
                 top_pct: float,
                 max_samples: int,
                 device: str = "cuda"):
        self.model_path  = model_path
        self.top_pct     = top_pct
        self.max_samples = max_samples

        self.tok = AutoTokenizer.from_pretrained(model_path, max_length=9000)
        self.model = (AutoModelForCausalLM
                      .from_pretrained(model_path)
                      .half().to(device).eval())

        self.layers = self._parse_layers(layers, len(self.model.model.layers))
        self.d_kv   = (self.model.config.hidden_size //
                       self.model.config.num_attention_heads *
                       self.model.config.num_key_value_heads)

    # ---------- abstract ------------------------------------------------
    @abc.abstractmethod
    def iter_examples(self):
        """
        Yield (context, positive_span, negative_span) strings.
        Stop anytime after `max_samples` total examples.
        """
        raise NotImplementedError

    # ---------- utilities ----------------------------------------------
    @staticmethod
    def _parse_layers(spec: str, n_total: int):
        if spec == "all":
            return list(range(n_total))
        if spec.startswith("last"):
            k = int(spec[4:])
            return list(range(n_total - k, n_total))
        return [int(x) for x in spec.split(',')]

    def _span_ids(self, text: str, sub: str):
        if sub.lower() not in text.lower():
            return None
        a, b = text.lower().index(sub.lower()), text.lower().index(sub.lower()) + len(sub)
        offs = self.tok(text,
                        return_offsets_mapping=True,
                        add_special_tokens=False)["offset_mapping"]
        return [i for i, (s, e) in enumerate(offs) if s >= a and e <= b]

    @torch.no_grad()
    def _layer_keys(self, text: str, idx):
        ids = self.tok(text, return_tensors="pt",
                       add_special_tokens=False).to(self.model.device)
        hidd = self.model(**ids,
                          use_cache=False,
                          output_hidden_states=True).hidden_states
        out = []
        for L in self.layers:
            h = hidd[L][0]                                    # (seq,d_model)
            k = self.model.model.layers[L].self_attn.k_proj(h)  # (seq,d_kv)
            out.append(k[idx].float())
        return out                                            # list(len_layers)

    @staticmethod
    def _build_proj(mat: torch.Tensor, pct: float):
        mat = (mat - mat.mean(0, keepdim=True)).double().cpu()
        C   = (mat.T @ mat) / mat.size(0)
        U, S, _ = torch.linalg.svd(C.float(), full_matrices=False)
        k  = (torch.cumsum(S, 0) / S.sum() < pct).sum().item() + 1
        return (U[:, :k] @ U[:, :k].T).to(torch.float16)      # (d,d)

    # ---------- public driver ------------------------------------------
    def run(self,
            pos_out: str,
            neg_out: str):
        pos_buf = [ [] for _ in self.layers ]
        neg_buf = [ [] for _ in self.layers ]

        for i, (ctx, pos_span, neg_span) in tqdm(enumerate(self.iter_examples()), total=min(len(self.data), self.max_samples)):
            if i >= self.max_samples:
                break

            idx_pos = self._span_ids(ctx, pos_span)
            idx_neg = self._span_ids(ctx, neg_span)

            if idx_pos:
                for buf, k in zip(pos_buf, self._layer_keys(ctx, idx_pos)):
                    buf.append(k)
            if idx_neg:
                for buf, k in zip(neg_buf, self._layer_keys(ctx, idx_neg)):
                    buf.append(k)

        # build PCA projectors
        pos_stack, neg_stack = [], []
        for L, (p, n) in enumerate(zip(pos_buf, neg_buf)):
            if not p or not n:
                raise RuntimeError(f"layer {self.layers[L]} missing samples")
            pos_stack.append(self._build_proj(torch.cat(p, 0), self.top_pct))
            neg_stack.append(self._build_proj(torch.cat(n, 0), self.top_pct))

        pos_proj = torch.stack(pos_stack)
        neg_proj = torch.stack(neg_stack)

        pathlib.Path(pos_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'layers': self.layers, 'proj': pos_proj.cpu()}, pos_out)
        torch.save({'layers': self.layers, 'proj': neg_proj.cpu()}, neg_out)

        print(f"✓ saved {pos_out}", tuple(pos_proj.shape))
        print(f"✓ saved {neg_out}",  tuple(neg_proj.shape))


