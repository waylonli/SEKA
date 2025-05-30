from __future__ import annotations
import abc, argparse, json, pathlib, torch
import os

from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
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
        # 1) buffers per layer, per head
        num_layers = len(self.layers)
        n_kv       = self.model.config.num_key_value_heads
        buf_H   = [[[] for _ in range(n_kv)] for _ in range(num_layers)]
        buf_Hp  = [[[] for _ in range(n_kv)] for _ in range(num_layers)]
        buf_Hn  = [[[] for _ in range(n_kv)] for _ in range(num_layers)]

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

                # extract keys: flat list of length num_layers*n_kv
                keys_H  = self.extract_keys(self.model, self.tokenizer, text_H,  idx_H,  self.layers, self.feature)
                keys_Hp = self.extract_keys(self.model, self.tokenizer, text_Hp, idx_Hp, self.layers, self.feature)
                keys_Hn = self.extract_keys(self.model, self.tokenizer, text_Hn, idx_Hn, self.layers, self.feature)

                # distribute into buffers
                for idx_flat, (k_H, k_Hp, k_Hn) in enumerate(zip(keys_H, keys_Hp, keys_Hn)):
                    L = idx_flat // n_kv
                    h = idx_flat %  n_kv
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
        all_pos_keys = {L: {h: [] for h in range(n_kv)} for L in range(num_layers)}
        all_neg_keys = {L: {h: [] for h in range(n_kv)} for L in range(num_layers)}
        norm_diffs = np.zeros((num_layers, n_kv))

        for L in tqdm(range(num_layers), desc="Computing Projectors", unit="layer"):
            Pp_heads, Pn_heads = [], []
            for h in range(n_kv):
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
                    # Pp = torch.eye(Pp.size(0), dtype=Pp.dtype, device=Pp.device)
                    # Pn = torch.eye(Pn.size(0), dtype=Pn.dtype, device=Pn.device)
                    skipped.append((self.layers[L], h, norm_value))
                    Pp = torch.zeros_like(Pp, dtype=Pp.dtype, device=Pp.device)
                    Pn = torch.zeros_like(Pn, dtype=Pp.dtype, device=Pp.device)

                else:
                    applied.append((self.layers[L], h, norm_value))

                Pp_heads.append(Pp)
                Pn_heads.append(Pn)

            pos_list.append(torch.stack(Pp_heads, dim=0))  # (H,d,d)
            neg_list.append(torch.stack(Pn_heads, dim=0))

        # stack layers → (num_layers, H, d, d)
        pos_proj = torch.stack(pos_list, dim=0)
        neg_proj = torch.stack(neg_list, dim=0)

        # save
        os.makedirs(output_dir, exist_ok=True)


        torch.save({'layers': self.layers, 'proj': pos_proj.cpu()}, os.path.join(output_dir,
                         f"{self.model_path.split('/')[-1]}_pos_proj_{self.feature}.pt") if self.feature else os.path.join(
                output_dir, f"{self.model_path.split('/')[-1]}_pos_proj.pt"))
        torch.save({'layers': self.layers, 'proj': neg_proj.cpu()}, os.path.join(output_dir,
                         f"{self.model_path.split('/')[-1]}_neg_proj_{self.feature}.pt") if self.feature else os.path.join(
                output_dir, f"{self.model_path.split('/')[-1]}_neg_proj.pt"))

        # summary
        print("\nProjection Summary:")
        if applied:
            print(f" ✔ Applied projection: {len(applied)}")
            for L, h, diff in applied:
                print(f"    • Layer {L}, Head {h}, Diff {diff:.2f}")
        if skipped:
            print(f" ✖ Skipped (identity): {len(skipped)}")
            for L, h, diff in skipped:
                print(f"    • Layer {L}, Head {h}, Diff {diff:.2f}")

        print(f"Saved positive projectors to {output_dir}, {tuple(pos_proj.shape)}")
        print(f"Saved negative projectors to {output_dir}, {tuple(neg_proj.shape)}")

        # Convert to numpy arrays
        all_pos_keys = {L: {h: np.concatenate(all_pos_keys[L][h], axis=0) for h in range(n_kv)} for L in
                        range(num_layers)}
        all_neg_keys = {L: {h: np.concatenate(all_neg_keys[L][h], axis=0) for h in range(n_kv)} for L in
                        range(num_layers)}

        # Visualize using PCA
        # self.visualize_key_shift(all_pos_keys, all_neg_keys, os.path.join(output_dir, f"kde_plot_{self.model_path.split('/')[-1]}"))

        # visualise the head norm differences
        # self.plot_norm_heatmap(norm_diffs, self.model_path, self.layers, output_dir)

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
            h_in = hiddens[L]
            attn = model.model.layers[L].self_attn

            if "qwen3" in model.model.layers[L].__class__.__name__.lower():
                h_in = model.model.layers[L].input_layernorm(h_in)
                if hasattr(attn, 'k_norm'):
                    h_in = model.model.layers[L].input_layernorm(h_in)
                    input_shape = h_in.shape[:-1]
                    dim_h = model.config.head_dim
                    # reshape into (seq_len, heads, head_dim)
                    k = attn.k_norm(attn.k_proj(h_in).view(*input_shape, -1, dim_h))[0]

            # h_in = hiddens[L][0]
            # attn = model.model.layers[L].self_attn

            # if "qwen3" in model.model.layers[L].__class__.__name__.lower():
            #     if hasattr(attn, 'k_norm'):
            #         n_kv = model.config.num_key_value_heads
            #         dim_h = model.config.head_dim
            #         # reshape into (seq_len, heads, head_dim)
            #         k = attn.k_proj(h_in).view(-1, n_kv, dim_h)
            #         k = attn.k_norm(k).view(-1, n_kv, dim_h)
            else:
                raise NotImplementedError(f"Unsupported model type: {model.__class__.__name__}. Currently only Qwen3 models are supported.")

            # check if k contains nan
            if torch.isnan(k).any():
                import pdb; pdb.set_trace()
                raise ValueError("k contains NaN values")

            k = phi(k.float(), feature).to(torch.float)
            # select only our tokens, and then return per-head slices
            k_sel = k[indices]  # (n_tokens, n_kv, dim_h)
            for h in range(k_sel.size(1)):
                result.append(k_sel[:, h, :])

        return result


    def visualize_key_shift(self, pos_keys, neg_keys, output_dir):
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times New Roman']
        num_layers = len(self.layers)
        n_kv = self.model.config.num_key_value_heads

        os.makedirs(output_dir, exist_ok=True)

        for L in tqdm(range(num_layers), desc="Visualizing Layers", unit="layer"):
            layer_dir = os.path.join(output_dir, f"Layer_{L+1}")
            os.makedirs(layer_dir, exist_ok=True)

            for h in range(n_kv):
                pos = pos_keys[L][h]
                neg = neg_keys[L][h]

                if pos.shape[0] != neg.shape[0] or np.allclose(pos, neg, atol=1e-6):
                    continue

                combined = np.concatenate([pos, neg], axis=0)
                pca = PCA(n_components=2)
                proj = pca.fit_transform(combined)
                pos_proj = proj[:len(pos)]
                neg_proj = proj[len(pos):]

                dx = pos_proj[:, 0] - neg_proj[:, 0]
                dy = pos_proj[:, 1] - neg_proj[:, 1]

                mean_dx = dx.mean()
                mean_dy = dy.mean()
                mean_start = neg_proj.mean(axis=0)

                plt.figure(figsize=(10, 8))
                plt.scatter(pos_proj[:, 0], pos_proj[:, 1], s=80, alpha=0.8, c="#006400",
                            label='Positive')  # Bigger, darker green
                plt.scatter(neg_proj[:, 0], neg_proj[:, 1], s=80, alpha=0.8, c="#FF6B6B",
                            label='Negative')  # Bigger, red

                plt.quiver(neg_proj[:, 0], neg_proj[:, 1], dx, dy,
                           angles='xy', scale_units='xy', scale=1, width=0.0032,
                           headwidth=6, headlength=8, alpha=0.6, color='grey')  # Thicker quiver

                plt.arrow(mean_start[0], mean_start[1], 2*mean_dx, 2*mean_dy,
                          head_width=0.8, head_length=0.8, color='#003366', linewidth=3.0,  # Bolder dark blue
                          length_includes_head=True, label='Mean shift')

                plt.xlabel("PCA Component 1", fontsize=38)
                plt.ylabel("PCA Component 2", fontsize=38)
                plt.title(f"Layer {self.layers[L]} - Head {h} (Pairwise Shift)", fontsize=40)
                plt.xticks([])
                plt.yticks([])
                plt.legend(loc='upper right', fontsize=24, frameon=False)
                plt.tight_layout()
                plt.savefig(os.path.join(layer_dir, f"Layer_{L+1}_Head_{h+1}_pca_pairwise_shift.pdf"), dpi=300)
                plt.close()

    @staticmethod
    def plot_norm_heatmap(norm_diffs, model_path, layers, output_dir):
        from matplotlib.colors import LinearSegmentedColormap
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times New Roman']
        # norm_diffs: shape (num_layers, n_kv). Transpose for heads as rows, layers as columns.
        data = norm_diffs.T  # Now shape is (n_kv, num_layers): y=heads, x=layers

        # Create custom green-red colormap
        cmap = LinearSegmentedColormap.from_list(
            "custom_green_red", ["#FF6B6B", "#fffbe0", "#006400"], N=256
        )

        plt.figure(figsize=(24, 6))
        im = plt.imshow(data, cmap=cmap, aspect='auto', origin='upper')

        plt.xlabel("Layer", fontsize=44)
        plt.ylabel("Head", fontsize=44)
        model_name = model_path.split('/')[-1] if ("chat" in model_path.split('/')[-1].lower() or "base" in model_path.split('/')[-1].lower()) else f"{model_path.split('/')[-1]}-Base"
        torch.save(norm_diffs, os.path.join(output_dir, f"norm_diffs_{model_name}.pt"))
        plt.title(model_name, fontsize=50, pad=14)
        plt.xticks(np.arange(data.shape[1]), [str(int(l)+1) for l in layers], fontsize=32)
        plt.yticks(np.arange(data.shape[0]), np.arange(data.shape[0])+1, fontsize=32)

        cbar = plt.colorbar(im, fraction=0.025, pad=0.02, aspect=40)
        cbar.set_label('Norm Value', fontsize=40)
        cbar.ax.tick_params(labelsize=40)

        plt.tight_layout(pad=2)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"hp_hn_norm_heatmap_{model_path.split('/')[-1]}.pdf"))
        plt.close()

