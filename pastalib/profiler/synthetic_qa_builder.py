from __future__ import annotations
import abc, argparse, json, pathlib, torch
import os

from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Gemma3ForCausalLM
)
from src.utils import phi, _parse_layers
import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib as mpl
warnings.filterwarnings("ignore")

torch.set_grad_enabled(False)

class ProjectionBuilderBase(abc.ABC):
    def __init__(
        self,
        model_path: str,
        data_path: str,
        layers: str,
        top_pct: float,
        feature: str | None,
        max_samples: int,
        min_diff: float,
        chat: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.data_path = data_path
        self.model_path = model_path
        self.top_pct = top_pct
        self.feature = feature
        with open(self.data_path) as f:
            self.max_samples = min(max_samples, len(f.readlines()))
        self.min_diff = min_diff
        self.chat = chat
        self.device = device

        if "qwen3" in model_path.lower(): 
            model_cls = AutoModelForCausalLM
        elif "gemma-3" in model_path.lower():
            model_cls = Gemma3ForCausalLM
        else:
            raise ValueError
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, max_length=9000)
        self.model = (
            model_cls
            .from_pretrained(model_path).to(device).eval()
        )
    
        self.layers = _parse_layers(layers, len(self.model.model.layers))

    def iter_examples(self):
        with open(self.data_path) as f:
            for i, line in enumerate(f):
                if i >= self.max_samples:
                    break
                yield json.loads(line)

    def get_triplets(self, ex: dict) -> list[tuple[str,str,str,str]]:
        return [
            (ex['context_1'], ex['question_1'], ex['answer_1'], ex['question_2']),
            (ex['context_2'], ex['question_2'], ex['answer_2'], ex['question_1'])
        ]

    def run(self, output_dir):
        # 1) buffers per layer, per head
        num_layers = len(self.layers)
        n_heads    = self.model.config.num_attention_heads
        buf_H   = [[[] for _ in range(n_heads)] for _ in range(num_layers)]
        buf_Hp  = [[[] for _ in range(n_heads)] for _ in range(num_layers)]
        buf_Hn  = [[[] for _ in range(n_heads)] for _ in range(num_layers)]

        pbar = tqdm(total=self.max_samples, desc="Extracting Keys", unit="ex")
        count = 0
        for ex in self.iter_examples():
            for ctx, rel_q, ans, irr_q in self.get_triplets(ex):
                # assemble texts
                if self.chat:
                    text_H = self.tokenizer.apply_chat_template([{"role":"user","content":f"Context: {ctx}"}], tokenize=False)
                    text_Hp= self.tokenizer.apply_chat_template([{"role":"user","content":f"Question: {rel_q}\nContext: {ctx}"}], tokenize=False)
                    text_Hn= self.tokenizer.apply_chat_template([{"role":"user","content":f"Question: {irr_q}\nContext: {ctx}"}], tokenize=False)
                else:
                    text_H, text_Hp, text_Hn = f"Context: {ctx} ", f"Question: {rel_q}\nContext: {ctx}", f"Question: {irr_q}\nContext: {ctx}"

                # find answer token indices
                idx_H  = self.span_token_indices(self.tokenizer, text_H,  ans)
                idx_Hp = self.span_token_indices(self.tokenizer, text_Hp, ans)
                idx_Hn = self.span_token_indices(self.tokenizer, text_Hn, ans)

                assert len(idx_H) == len(idx_Hp) == len(idx_Hn), f"Indices mismatch: {len(idx_H)}, {len(idx_Hp)}, {len(idx_Hn)}"

                if not (idx_H and idx_Hp and idx_Hn):
                    continue

                # extract keys: flat list of length num_layers*n_heads
                keys_H  = self.extract_keys(self.model, self.tokenizer, text_H,  idx_H,  self.layers, self.feature)
                keys_Hp = self.extract_keys(self.model, self.tokenizer, text_Hp, idx_Hp, self.layers, self.feature)
                keys_Hn = self.extract_keys(self.model, self.tokenizer, text_Hn, idx_Hn, self.layers, self.feature)

                # distribute into buffers
                for idx_flat, (k_H, k_Hp, k_Hn) in enumerate(zip(keys_H, keys_Hp, keys_Hn)):
                    L = idx_flat // n_heads
                    h = idx_flat %  n_heads
                    buf_H[L][h].append(k_H)
                    buf_Hp[L][h].append(k_Hp)
                    buf_Hn[L][h].append(k_Hn)

            count += 1
            pbar.update(1)

            if count >= self.max_samples:
                break
        pbar.close()

        # 2) compute per-layer, per-head projectors
        pos_list, neg_list = [], []
        applied, skipped = [], []
        all_pos_keys = {L: {h: [] for h in range(n_heads)} for L in range(num_layers)}
        all_neg_keys = {L: {h: [] for h in range(n_heads)} for L in range(num_layers)}
        norm_diffs = np.zeros((num_layers, n_heads))

        for L in tqdm(range(num_layers), desc="Computing Projectors", unit="layer"):
            # Pp_heads, Pn_heads = [], []
            for h in range(n_heads):
                H_mat  = torch.cat(buf_H[L][h],  0).double().to(self.device)
                Hp_mat = torch.cat(buf_Hp[L][h], 0).double().to(self.device)
                Hn_mat = torch.cat(buf_Hn[L][h], 0).double().to(self.device)

                # Store positive and negative embeddings
                all_pos_keys[L][h].append(Hp_mat.cpu().detach().numpy())
                all_neg_keys[L][h].append(Hn_mat.cpu().detach().numpy())

                # neutral PCA → P0
                C0 = (H_mat.T @ H_mat) / H_mat.size(0)
                U0, S0, _ = torch.linalg.svd(C0.float(), full_matrices=False)
                k0 = (S0.cumsum(0)/S0.sum() < self.top_pct).sum().item() + 1
                P0 = (U0[:, :k0] @ U0[:, :k0].T).to(torch.float)

                # cross-cov → Pp, Pn
                Omega_p = (H_mat.T @ Hp_mat) / H_mat.size(0)
                Omega_n = (H_mat.T @ Hn_mat) / H_mat.size(0)
                Up, Sp, _ = torch.linalg.svd(Omega_p.float(), full_matrices=False)
                Un, Sn, _ = torch.linalg.svd(Omega_n.float(), full_matrices=False)
                kp = (Sp.cumsum(0)/Sp.sum() < self.top_pct).sum().item() + 1
                kn = (Sn.cumsum(0)/Sn.sum() < self.top_pct).sum().item() + 1
                Pp = (Up[:, :kp] @ Up[:, :kp].T).to(torch.float)
                Pn = (Un[:, kn:] @ Un[:, kn:].T).to(torch.float)

                norm_value = (torch.norm(Hp_mat - Hn_mat) / len(Hp_mat)).item()
                norm_diffs[L, h] = norm_value

                # decide
                if norm_value < self.min_diff:
                    skipped.append((self.layers[L], h, norm_value))
                    Pp = torch.zeros_like(Pp, dtype=Pp.dtype, device=Pp.device)
                    Pn = torch.zeros_like(Pn, dtype=Pp.dtype, device=Pp.device)

                else:
                    applied.append((self.layers[L], h, norm_value))

        # save
        os.makedirs(output_dir, exist_ok=True)

        # summary
        print("\nProjection Summary:")
        if applied:
            print(f" ✔ Applied projection: {len(applied)}")
            head_config = {}
            for L, h, diff in applied:
                print(f"    • Layer {L}, Head {h}, Diff {diff:.2f}")
                layer_key = str(L)
                head_config.setdefault(layer_key, []).append(h)
            with open(os.path.join(output_dir, f"{self.model_path.split('/')[-1]}_head_config.json"), 'w') as f:
                json.dump(head_config, f, indent=4)
        if skipped:
            print(f" ✖ Skipped (identity): {len(skipped)}")
            for L, h, diff in skipped:
                print(f"    • Layer {L}, Head {h}, Diff {diff:.2f}")


    @staticmethod
    def span_token_indices(tokenizer, text: str, sub: str) -> list[int] | None:
        low, sub_low = text.lower(), sub.lower()
        if sub_low not in low:
            return None
        start = low.index(sub_low)
        end = start + len(sub_low)
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        return [i for i, (s, e) in enumerate(enc.offset_mapping) if s >= start and e <= end]

    @staticmethod
    def extract_keys(model, tokenizer, text: str, indices: list[int], layers: list[int], feature: str) -> list[
        torch.Tensor]:
        result: list[torch.Tensor] = []

        def _hook(_, args):
            attn_out = args[0][0] # (seq_len, hidden)
            dim_h = model.config.head_dim
            seq_len = attn_out.shape[0]
            attn_out = attn_out.view(seq_len, -1, dim_h)  # (seq_len, n_heads, h_dim)
            
            if torch.isnan(attn_out).any():
                import pdb; pdb.set_trace()
                raise ValueError("h_out contains NaN values")
            
            attn_out = phi(attn_out.float(), feature).to(torch.float)
            
            # select only our tokens, and then return per-head slices
            attn_sel = attn_out[indices]  # (n_tokens, n_h, dim_h)
            for h in range(attn_sel.size(1)):
                result.append(attn_sel[:, h, :])
        
        hooks = [model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(_hook) for L in layers]
        
        inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).to(model.device)
        outputs = model(**inputs)
        
        for hook in hooks:
            hook.remove()

        return result
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--layers', default='all')
    parser.add_argument('--top_pct', type=float, default=0.9)
    parser.add_argument('--feature', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--min_diff', type=float, default=2)
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    builder = ProjectionBuilderBase(
        model_path=args.model,
        data_path=args.data,
        layers=args.layers,
        top_pct=args.top_pct,
        feature=args.feature,
        max_samples=args.max_samples,
        min_diff=args.min_diff,
        chat=args.chat
    )
    builder.run(args.output_dir)

