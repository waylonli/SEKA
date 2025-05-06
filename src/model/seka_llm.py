from __future__ import annotations
import torch, types
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import encode_with_markers, _parse_layers, _load_proj


class SEKALLM:
    """
    Basic use
    ---------
    ks = SEKALLM("pretrained/qwen2-1.5b-chat")
    ids, mask = utils.encode_with_markers("Write code. *Respond in JSON*.", ks.tok)

    # steer with positive projector only
    ks.attach_projection(pos_pt="pos_proj.pt", layers="last4",
                         steer_mask_tensor=mask, amplify_pos=2.0)

    # steer with both positive and negative projectors
    ks.attach_projection(pos_pt="pos_proj.pt", neg_pt="neg_proj.pt",
                         layers="last4", steer_mask_tensor=mask,
                         amplify_pos=2.0, amplify_neg=0.3)
    """

    # ───────────── init ────────────────────────────────────────────────
    def __init__(self,
                 model_or_path: str,
                 *,
                 device: str | None = "auto",
                 marker_start: str = "*",
                 marker_end: str | None = None,
                 pos_pt: str = None,
                 neg_pt: str | None = None,
                 layers: str = "last4",
                 amplify_pos: float = 0.8,
                 amplify_neg: float = 0.2,
                 feature_function: str | None = None,
                 **hf_kwargs
                 ):
        if device == "auto":
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

        self.name_or_path = f"SEKA-{model_or_path}"
        self.tok = AutoTokenizer.from_pretrained(model_or_path, **hf_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
                         model_or_path, use_cache=False, **hf_kwargs
                     ).to(device).eval()

        self.m_start = marker_start
        self.m_end = marker_start if marker_end is None else marker_end

        self.pos_pt = pos_pt
        self.neg_pt = neg_pt
        self.layers = layers
        self.amplify_pos = amplify_pos
        self.amplify_neg = amplify_neg

        if feature_function is None:
            if "_tanh" in pos_pt:
                self.feature_function = "tanh"
            elif "_elu" in pos_pt:
                self.feature_function = "elu"
            elif "_squared" in pos_pt:
                self.feature_function = "squared-exponential"
            else:
                self.feature_function = None
        else:
            self.feature_function = feature_function

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        # transparently expose everything from the HF model
        object.__setattr__(self, "__getattr__", lambda n: getattr(self.model, n))

    # ───────────── public helpers ──────────────────────────────────────
    @property
    def device(self):  # convenience
        return self.model.device

    def generate(self,
                 ids: torch.LongTensor | str,
                 steer: bool = True,
                 steer_mask: torch.Tensor | None = None,
                 attention_mask: torch.Tensor | None = None,
                 return_raw: bool = False,
                 **gen_kw) -> str:

        # TODO Seems there are some bugs for the batch decoding, check steer mask and projection multiplication
        if isinstance(ids, (str, list)):
            ids, steer_mask, attention_mask = encode_with_markers(ids, self.tok, self.m_start, self.m_end)
            ids = ids.to(self.device)
            steer_mask = steer_mask.to(self.device)
            attention_mask = attention_mask.to(self.device)
        elif isinstance(ids, torch.Tensor):
            if steer:
                assert steer_mask is not None, "steer_mask must be provided if ids is a tensor"
                steer_mask = steer_mask.to(self.device)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask
            attention_mask = attention_mask.to(self.device)

        # -------- optional steering --------
        if steer:
            self.attach_projection(steer_mask_tensor=steer_mask, silence=True)
        else:
            self.remove_projection()

        out = self.model.generate(
            ids,
            **gen_kw
        )

        if steer:
            self.remove_projection()

        if return_raw:
            return out

        generated = []
        for i in range(ids.size(0)):
            generated.append(
                self.tok.decode(out[i, ids.size(1):], skip_special_tokens=True)
            )

        return generated[0] if len(generated) == 1 else generated



    # ───────────── steering control ────────────────────────────────────
    def attach_projection(self,
                          steer_mask_tensor=None,
                          pos_pt=None,
                          neg_pt=None,
                          layers=None,
                          amplify_pos=None,
                          amplify_neg=None,
                          feature_function=None,
                          silence=False):

        self.remove_projection()  # clear old hooks

        # ----- defaults -------------------------------------------------
        pos_pt = self.pos_pt if pos_pt is None else pos_pt
        neg_pt = self.neg_pt if neg_pt is None else neg_pt
        layers = self.layers if layers is None else layers
        amplify_pos = self.amplify_pos if amplify_pos is None else amplify_pos
        amplify_neg = self.amplify_neg if amplify_neg is None else amplify_neg
        if feature_function is None:
            if self.feature_function is None:
                if "_tanh" in pos_pt:
                    feature_function = "tanh"
                elif "_elu" in pos_pt:
                    feature_function = "elu"
                elif "_squared" in pos_pt:
                    feature_function = "squared-exponential"
                else:
                    feature_function = None
            else:
                feature_function = self.feature_function


        dev, n_layers = self.device, len(self.model.model.layers)
        dtype = torch.float32

        # ----- load projectors -----------------------------------------
        file_layers, P_pos_stack = _load_proj(pos_pt, dev)
        sel_layers = _parse_layers(layers, n_layers)
        if file_layers is not None:
            P_pos = {L: P_pos_stack[i].T.contiguous() for i, L in enumerate(file_layers)}
        else:
            P_pos = {L: P_pos_stack[0].T.contiguous() for L in sel_layers}

        if neg_pt:
            f_layers, P_neg_stack = _load_proj(neg_pt, dev)
            if f_layers is not None:
                P_neg = {L: P_neg_stack[i].T.contiguous() for i, L in enumerate(f_layers)}
            else:
                P_neg = {L: P_neg_stack[0].T.contiguous() for L in sel_layers}
        else:
            P_neg = None

        if steer_mask_tensor is not None:
            steer_mask_tensor = steer_mask_tensor.unsqueeze(0) if steer_mask_tensor.dim() == 1 else steer_mask_tensor
            m_dev = steer_mask_tensor.to(dev)
        else:
            m_dev = None

        # ----- kernel helpers ------------------------------------------
        def phi(x):
            if feature_function is None:
                return x
            if feature_function == "squared-exponential":
                return torch.exp(-x.pow(2) / 2)
            if feature_function == "tanh":
                return torch.tanh(x)
            if feature_function == "elu":
                return torch.where(x >= 0, x, torch.exp(x) - 1)
            raise ValueError(f"unknown feature_function {feature_function}")

        def phi_inv(x):
            if feature_function is None:
                return x
            eps = torch.full_like(x, 1e-4)
            if feature_function == "squared-exponential":
                return -torch.log(torch.clamp(x, min=eps)) * 2
            if feature_function == "tanh":
                x = torch.clamp(x, -1 + eps, 1 - eps)
                return torch.atanh(x)
            if feature_function == "elu":
                pos = torch.clamp(x, min=0)
                neg = torch.clamp(x, max=0)
                return pos + torch.log(neg + 1)
            return x

        # ----- register per‑layer hooks --------------------------------
        for L in sel_layers:
            Pp = P_pos[L].to(dtype)
            Pn = P_neg[L].to(dtype) if P_neg else None

            attn_layer = self.model.model.layers[L].self_attn
            hook_module = attn_layer.k_norm if hasattr(attn_layer, "k_norm") else attn_layer.k_proj

            def _hook(_, __, k_in,
                      m=m_dev, P_pos=Pp, P_neg=Pn,
                      g_pos=amplify_pos, g_neg=amplify_neg):

                # ---------- unify shape ---------------------------------
                four_d = (k_in.dim() == 4)  # (B,T,H,D)
                if four_d:
                    B, T, H, D = k_in.shape
                    k_flat = k_in.reshape(B, T, H * D)  # → (B,T,d_kv)
                else:
                    B, T, _ = k_in.shape
                    k_flat = k_in  # already flat

                # ---------- kernel map ---------------------------------
                k_feat = phi(k_flat.float())  # φ(k)

                # ---------- no mask: steer whole sequence --------------
                if m is None:
                    k_new = phi_inv(k_feat).to(k_in.dtype)
                    return (k_new.reshape(B, T, H, D) if four_d else k_new)

                # ---------- shape check --------------------------------
                if m.shape != (B, T):
                    return k_in

                # ---------- per‑batch edit -----------------------------
                for b in range(B):
                    mb = m[b]
                    pos_idx = mb.nonzero().squeeze(-1)
                    neg_idx = (~mb).nonzero().squeeze(-1)

                    if pos_idx.numel():
                        kb = k_feat[b, pos_idx, :]
                        kb = kb + g_pos * (kb @ P_pos)
                        k_flat[b, pos_idx, :] = phi_inv(kb).to(k_in.dtype)

                    if P_neg is not None and neg_idx.numel():
                        kb = k_feat[b, neg_idx, :]
                        kb = kb + g_neg * (kb @ P_neg)
                        k_flat[b, neg_idx, :] = phi_inv(kb).to(k_in.dtype)

                # ---------- restore original rank ----------------------
                k_out = k_flat.reshape(B, T, H, D) if four_d else k_flat

                return k_out

            self._hooks.append(hook_module.register_forward_hook(_hook, prepend=True))

        if not silence:
            tag = "pos+neg" if P_neg else "pos‑only"
            print(
                f"[Key‑Steering] {tag}, layers {sel_layers}, "
                f"g+={amplify_pos}, g-={amplify_neg if P_neg else 'n/a'}, "
                f"kernel={feature_function or 'linear'}"
            )

    def remove_projection(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def eval(self):
        pass

    def train(self):
        pass

    def to(self, device):
        pass
