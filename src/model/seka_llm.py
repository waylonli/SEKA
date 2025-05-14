from __future__ import annotations
import torch, types
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import encode_with_markers, _parse_layers, _load_proj, phi, phi_inv

class SEKALLM:
    # ───────────── init ────────────────────────────────────────────────
    def __init__(self,
                 model_or_path: str,
                 *,
                 device: str | None = "auto",
                 marker_start: str = "**",
                 marker_end: str | None = None,
                 pos_pt: str = None,
                 neg_pt: str | None = None,
                 layers: str = "last10",
                 amplify_pos: float = 0.8,
                 amplify_neg: float = 0.2,
                 feature_function: str | None = None,
                 **hf_kwargs
                 ):
        # ----- device selection -----
        if device == "auto":
            device = ("cuda" if torch.cuda.is_available()
                      else "mps"  if torch.backends.mps.is_available()
                      else "cpu")

        multi_gpu = torch.cuda.device_count() > 1 and str(device).startswith("cuda")

        # ----- HF objects -----
        self.name_or_path = f"SEKA-{model_or_path}"
        self.tok = AutoTokenizer.from_pretrained(model_or_path, **hf_kwargs)

        if multi_gpu:
            # shard across all GPUs
            self.model = AutoModelForCausalLM.from_pretrained(
                model_or_path,
                device_map="auto",
                use_cache=False,
                **hf_kwargs
            ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_or_path,
                use_cache=False,
                **hf_kwargs
            ).to(device).eval()

        # markers & steering defaults
        self.m_start, self.m_end = marker_start, (marker_start if marker_end is None else marker_end)
        self.pos_pt, self.neg_pt = pos_pt,   neg_pt
        self.layers = layers
        self.amplify_pos, self.amplify_neg = amplify_pos, amplify_neg

        if feature_function is None:
            if "_tanh"     in str(pos_pt): self.feature_function = "tanh"
            elif "_elu"    in str(pos_pt): self.feature_function = "elu"
            elif "_squared" in str(pos_pt): self.feature_function = "squared-exponential"
            else:                          self.feature_function = None
        else:
            self.feature_function = feature_function

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        # expose everything from the HF model
        object.__setattr__(self, "__getattr__", lambda n: getattr(self.model, n))

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
            # check decoded ids
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

        if "attention_mask" not in gen_kw and attention_mask is not None:
            gen_kw["attention_mask"] = attention_mask

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

    # ───────────── steering ────────────────────────────────────────────
    def attach_projection(
        self,
        steer_mask_tensor=None,
        pos_pt=None,
        neg_pt=None,
        layers=None,
        amplify_pos=None,
        amplify_neg=None,
        feature_function=None,
        silence=False
    ):
        self.remove_projection()

        # defaults
        pos_pt         = self.pos_pt         if pos_pt  is None else pos_pt
        neg_pt         = self.neg_pt         if neg_pt  is None else neg_pt
        layers         = self.layers         if layers  is None else layers
        amplify_pos    = self.amplify_pos    if amplify_pos is None else amplify_pos
        amplify_neg    = self.amplify_neg    if amplify_neg is None else amplify_neg
        feature_function = self.feature_function if feature_function is None else feature_function

        n_layers = len(self.model.model.layers)
        dtype = torch.float32

        # load projections on first device (cheap to move later)
        first_dev = self.device
        file_layers, P_pos_stack = _load_proj(pos_pt, first_dev)  # → (L_sel,H,d,d)
        sel_layers = _parse_layers(layers, n_layers)
        if P_pos_stack.ndim != 4:
            raise ValueError("Expected 4‑D pos_proj")
        P_pos = {layer: P_pos_stack[i].to(dtype) for i, layer in enumerate(sel_layers)}

        P_neg = None
        if neg_pt:
            _, P_neg_stack = _load_proj(neg_pt, first_dev)
            if P_neg_stack.ndim != 4:
                raise ValueError("Expected 4‑D neg_proj")
            P_neg = {layer: P_neg_stack[i].to(dtype) for i, layer in enumerate(sel_layers)}

        # steering mask (will still be relocated inside hook)
        m_dev = (steer_mask_tensor if steer_mask_tensor is None
                 else steer_mask_tensor.unsqueeze(0) if steer_mask_tensor.dim() == 1
                 else steer_mask_tensor).to(first_dev) if steer_mask_tensor is not None else None

        # pick correct root (handles DataParallel etc.)
        root = self.model.module if hasattr(self.model, "module") else self.model

        # register hooks
        for L in sel_layers:
            Pp_layer = P_pos[L]       # (H,d,d)
            Pn_layer = P_neg[L] if P_neg else None
            attn = root.model.layers[L].self_attn
            mod  = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj

            def _hook(_, __, k_in,
                      m=m_dev, Pp=Pp_layer, Pn=Pn_layer,
                      gp=amplify_pos, gn=amplify_neg):
                # move tensors to the layer’s device
                dev = k_in.device
                m  = None if m is None else m.to(dev)
                Pp = Pp.to(dev)
                Pn = None if Pn is None else Pn.to(dev)

                if Pp.sum() == 0:
                    return k_in

                B, T, H, D = k_in.shape
                k_feat = phi(k_in.float(), feature_function)

                if m is None or m.shape != (B, T) or m.sum() == 0:
                    return phi_inv(k_feat, feature_function).to(k_in.dtype)

                k_out = k_feat.clone()
                for b in range(B):
                    idx = m[b].nonzero(as_tuple=True)[0]
                    if idx.numel():
                        for h in range(H):
                            kb = k_feat[b, idx, h]
                            if Pn is not None:
                                kb = kb + ((gp * (kb @ Pp[h]) + gn * (kb @ Pn[h])) / 2)
                            else:
                                kb = kb + gp * (kb @ Pp[h])
                            k_out[b, idx, h] = phi_inv(kb, feature_function).to(k_in.dtype)
                return k_out.to(k_in.dtype)

            self._hooks.append(mod.register_forward_hook(_hook))

        if not silence:
            print(f"✅ Steering hooks attached on layers {sel_layers}")

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
