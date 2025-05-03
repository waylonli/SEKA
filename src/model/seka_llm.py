# Hugging‑Face‑style wrapper that supports both a *positive* (relevant)
# and an optional *negative* (irrelevant) key‑projection.
#
# Positive projector  : boosts keys on the *‑starred‑* tokens.
# Negative projector  : suppresses keys everywhere else (if supplied).

from __future__ import annotations
import torch, types
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import encode_with_markers, _parse_layers, _load_proj


class SEKALLM:
    """
    Basic use
    ---------
    ks = KeySteeredLM("qwen2-1.5b-chat")
    ids, mask = encode_with_mask("Write code. *Respond in JSON*.", ks.tok)

    # steer with positive projector only
    ks.attach_projection(rel_pt="rel_proj.pt", layers="last4",
                         mask_tensor=mask, amplify_pos=2.0)

    # steer with both positive and negative projectors
    ks.attach_projection(rel_pt="rel_proj.pt", irrel_pt="irrel_proj.pt",
                         layers="last4", mask_tensor=mask,
                         amplify_pos=2.0, amplify_neg=0.3)
    """

    # ───────────── init ────────────────────────────────────────────────
    def __init__(self,
                 model_or_path: str,
                 *,
                 device: str | None = "auto",
                 marker_start: str = "*",
                 marker_end: str | None = None,
                 **hf_kwargs):
        if device == "auto":
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

        self.tok   = AutoTokenizer.from_pretrained(model_or_path, **hf_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
                         model_or_path, use_cache=False, **hf_kwargs
                     ).to(device).eval()

        self.m_start = marker_start
        self.m_end   = marker_start if marker_end is None else marker_end

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        # transparently expose everything from the HF model
        object.__setattr__(self, "__getattr__", lambda n: getattr(self.model, n))

    # ───────────── public helpers ──────────────────────────────────────
    @property
    def device(self):  # convenience
        return self.model.device

    def generate(self,
                 ids: torch.LongTensor | str,
                 mask_tensor: torch.Tensor | None = None,
                 **gen_kw) -> str:
        if isinstance(ids, str):
            ids, auto_mask = encode_with_markers(txt, self.tok, self.m_start, self.m_end)
            mask_tensor = auto_mask if mask_tensor is None else mask_tensor

        ids = ids.to(self.device)

        out = self.model.generate(
                  ids,
                  max_new_tokens=gen_kw.pop("max_new_tokens", 128),
                  pad_token_id=self.tok.eos_token_id,
                  do_sample=False, temperature=0.0,
                  **gen_kw
              )
        return self.tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)

    # ───────────── steering control ────────────────────────────────────
    def attach_projection(self,
                          *,
                          rel_pt: str,
                          irrel_pt: str | None = None,
                          layers: str = "last4",
                          mask_tensor: torch.Tensor | None = None,
                          amplify_pos: float = 0.8,
                          amplify_neg: float = 0.2):
        """
        Parameters
        ----------
        rel_pt      path to the positive projector (required)
        irrel_pt    path to negative projector; if None → positive‑only
        layers      "last4" | "all" | "0,4,19" ...
        mask_tensor BoolTensor(seq,) where True marks *highlighted* tokens
        amplify_pos >1 boosts relevant keys on mask True positions
        amplify_neg <1 suppresses irrelevant keys on mask False positions
        """
        self.remove_projection()                 # clear old hooks first

        dev, n_layers = self.device, len(self.model.model.layers)
        dtype = torch.float32                    # keep projections in fp32

        # ---------- load positive projector ----------
        file_layers, P_pos_stack = _load_proj(rel_pt, dev)
        sel_layers = _parse_layers(layers, n_layers)
        if file_layers is not None:
            P_pos = {L: P_pos_stack[i].T.contiguous() for i, L in enumerate(file_layers)}
        else:
            P_pos = {L: P_pos_stack[0].T.contiguous() for L in sel_layers}

        # ---------- load negative projector (optional) ----------
        if irrel_pt:
            f_layers, P_neg_stack = _load_proj(irrel_pt, dev)
            if f_layers is not None:
                P_neg = {L: P_neg_stack[i].T.contiguous() for i, L in enumerate(f_layers)}
            else:
                P_neg = {L: P_neg_stack[0].T.contiguous() for L in sel_layers}
        else:
            P_neg = None

        m_dev = mask_tensor.to(dev) if mask_tensor is not None else None

        # ---------- register hooks ----------
        for L in sel_layers:
            Pp = P_pos[L].to(dtype)
            Pn = P_neg[L].to(dtype) if P_neg else None
            k_lin = self.model.model.layers[L].self_attn.k_proj

            def _hook(_, __, k_out,
                      m=m_dev, P_pos=Pp, P_neg=Pn,
                      γp=amplify_pos, γn=amplify_neg):
                # no mask → whole sequence treated as positive only
                if m is None:
                    return k_out + γp * (k_out @ P_pos)

                if k_out.shape[1] != m.numel():
                    return k_out

                sel = m.nonzero(as_tuple=False).squeeze(-1)          # positive
                non_sel = (~m).nonzero(as_tuple=False).squeeze(-1)   # negative

                if sel.numel():
                    k_sel = k_out[:, sel, :]
                    k_out[:, sel, :] = k_sel + γp * (k_sel @ P_pos)

                if P_neg is not None and non_sel.numel():
                    k_ns = k_out[:, non_sel, :]
                    k_out[:, non_sel, :] = k_ns + γn * (k_ns @ P_neg)

                return k_out

            self._hooks.append(k_lin.register_forward_hook(_hook, prepend=True))

        tag = ("pos+neg" if P_neg is not None else "pos‑only")
        print(f"[Key‑Steering] {tag}, layers {sel_layers}, "
              f"γ+={amplify_pos}, γ‑={amplify_neg if P_neg else 'n/a'}")

    def remove_projection(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
