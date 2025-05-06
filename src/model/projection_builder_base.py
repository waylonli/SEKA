from __future__ import annotations
import abc, argparse, json, pathlib, torch
import os

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import phi, _parse_layers

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

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, max_length=9000)
        self.model = (
            AutoModelForCausalLM
            .from_pretrained(model_path).to(device).eval()
        )
        self.layers = _parse_layers(layers, len(self.model.model.layers))

    @abc.abstractmethod
    def iter_examples(self):
        """Yield raw examples from data source."""
        ...

    @abc.abstractmethod
    def get_triplets(self, example: dict) -> list[tuple[str, str, str, str]]:
        """For a given example, return list of (context, rel_q, ans, irr_q) tuples."""
        ...

    def run(self, output_dir):
        buf_H = [[] for _ in self.layers]
        buf_Hp = [[] for _ in self.layers]
        buf_Hn = [[] for _ in self.layers]

        total = self.max_samples
        pbar = tqdm(total=total, desc='Processing')
        count = 0
        for ex in self.iter_examples():
            if count >= self.max_samples:
                break
            for ctx, rel_q, ans, irr_q in self.get_triplets(ex):
                if self.chat:
                    text_H = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": ctx}], tokenize=False
                    )
                    text_Hp = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": f"{ctx} {rel_q}"}], tokenize=False
                    )
                    text_Hn = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": f"{ctx} {irr_q}"}], tokenize=False
                    )
                else:
                    text_H, text_Hp, text_Hn = ctx, f"{ctx} {rel_q}", f"{ctx} {irr_q}"

                idx_H = self.span_token_indices(self.tokenizer, text_H, ans)
                idx_Hp = self.span_token_indices(self.tokenizer, text_Hp, ans)
                idx_Hn = self.span_token_indices(self.tokenizer, text_Hn, ans)
                if not (idx_H and idx_Hp and idx_Hn):
                    continue

                keys_H = self.extract_keys(self.model, self.tokenizer, text_H, idx_H, self.layers, self.feature)
                keys_Hp = self.extract_keys(self.model, self.tokenizer, text_Hp, idx_Hp, self.layers, self.feature)
                keys_Hn = self.extract_keys(self.model, self.tokenizer, text_Hn, idx_Hn, self.layers, self.feature)

                for i in range(len(self.layers)):
                    buf_H[i].append(keys_H[i])
                    buf_Hp[i].append(keys_Hp[i])
                    buf_Hn[i].append(keys_Hn[i])
                count += 1
                pbar.update(1)
                if count >= self.max_samples:
                    break
            if count >= self.max_samples:
                break
        pbar.close()

        pos_proj, neg_proj = [], []
        for i in range(len(self.layers)):
            H_mat = torch.cat(buf_H[i], 0).double().cpu()
            Hp_mat = torch.cat(buf_Hp[i], 0).double().cpu()
            Hn_mat = torch.cat(buf_Hn[i], 0).double().cpu()

            C0 = (H_mat.T @ H_mat) / H_mat.size(0)
            U0, S0, _ = torch.linalg.svd(C0.float(), full_matrices=False)
            k0 = (S0.cumsum(0) / S0.sum() < self.top_pct).sum().item() + 1
            P0 = (U0[:, :k0] @ U0[:, :k0].T).to(torch.float16)

            Omega_p = (H_mat.T @ Hp_mat) / H_mat.size(0)
            Omega_n = (H_mat.T @ Hn_mat) / H_mat.size(0)

            Up, Sp, _ = torch.linalg.svd(Omega_p.float(), full_matrices=False)
            Un, Sn, _ = torch.linalg.svd(Omega_n.float(), full_matrices=False)

            kp = (Sp.cumsum(0) / Sp.sum() < self.top_pct).sum().item() + 1
            kn = (Sn.cumsum(0) / Sn.sum() < self.top_pct).sum().item() + 1

            Pp = (Up[:, :kp] @ Up[:, :kp].T).to(torch.float16)
            Pn = (Un[:, -kn:] @ Un[:, -kn:].T).to(torch.float16)

            if torch.norm(Pp - P0) < self.min_diff:
                Pp = P0
            if torch.norm(Pn - P0) < self.min_diff:
                Pn = P0

            pos_proj.append(Pp)
            neg_proj.append(Pn)

        pos_tensor = torch.stack(pos_proj)
        neg_tensor = torch.stack(neg_proj)

        os.makedirs(output_dir, exist_ok=True)
        torch.save({'layers': self.layers, 'proj': pos_tensor.cpu()}, os.path.join(output_dir, f"{self.model_path.split('/')[-1]}_pos_proj.pt"))
        torch.save({'layers': self.layers, 'proj': neg_tensor.cpu()}, os.path.join(output_dir, f"{self.model_path.split('/')[-1]}_neg_proj.pt"))

        print(f"Saved positive projectors to {output_dir}, {tuple(pos_tensor.shape)}")
        print(f"Saved negative projectors to {output_dir}, {tuple(neg_tensor.shape)}")

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
        inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).to(model.device)
        outputs = model(**inputs, use_cache=False, output_hidden_states=True)
        hiddens = outputs.hidden_states
        result: list[torch.Tensor] = []
        for L in layers:
            h_in = hiddens[L][0]
            attn = model.model.layers[L].self_attn
            if hasattr(attn, 'k_norm'):
                n_kv = model.config.num_key_value_heads
                dim_h = model.config.hidden_size // model.config.num_attention_heads
                k = attn.k_proj(h_in).view(-1, n_kv, dim_h)
                k = attn.k_norm(k).view(-1, n_kv * dim_h)
            else:
                k = attn.k_proj(h_in)
            # check if k contains nan
            if torch.isnan(k).any():
                import pdb; pdb.set_trace()
                raise ValueError("k contains NaN values")

            k = phi(k.float(), feature).to(torch.float16)
            result.append(k[indices])
        return result
