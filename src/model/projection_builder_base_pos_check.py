# src/model/projection_builder_base.py
from __future__ import annotations
import abc, torch
import os

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
            save_svd: bool = False,
            save_traditional: bool = True,
            attention_threshold: float = 1.2, # NEW: Threshold for attention increase
    ):
        self.data_path = data_path
        self.model_path = model_path
        self.top_pct = top_pct
        self.feature = feature
        self.max_samples = max_samples
        self.min_diff = min_diff
        self.chat = chat
        self.device = device
        self.save_svd = save_svd
        self.save_traditional = save_traditional
        self.attention_threshold = attention_threshold # NEW: Store threshold

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, max_length=9000)
        self.model = (
            AutoModelForCausalLM
            .from_pretrained(model_path).to(device).eval()
        )
        if not "gemma3" in self.model.__class__.__name__.lower():
            self.layers = _parse_layers(layers, len(self.model.model.layers))
        else:
            self.layers = _parse_layers(layers, len(self.model.language_model.model.layers))

    @abc.abstractmethod
    def iter_examples(self):
        """Yield raw examples from data source."""
        ...

    @abc.abstractmethod
    def get_triplets(self, example: dict) -> list[tuple[str, str, str]]:
        """For a given example, return list of (context, rel_q, ans) tuples."""
        ...
        
    def assemble_texts(self, ctx: str, rel_q: str):
        if self.chat:
            text_H = self.tokenizer.apply_chat_template([{"role": "user", "content": f"Context: {ctx}"}],
                                                        tokenize=False)
            text_Hp = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"Question: {rel_q}\nContext: {ctx}"}], tokenize=False)
        else:
            text_H, text_Hp = f"Context: {ctx} ", f"Question: {rel_q}\nContext: {ctx}"

        return text_H, text_Hp

    def _check_attention_increase(self, text_H, text_Hp, indices_H, indices_Hp):
        """
        NEW: Checks if attention on the answer span increases from H to Hp.
        Returns a tuple: (is_valid, model_outputs_H, model_outputs_Hp)
        The model outputs are returned to avoid re-computing them later.
        """
        # 1. Tokenize and get model outputs including attentions
        inputs_H = self.tokenizer(text_H, return_tensors='pt', add_special_tokens=False).to(self.device)
        outputs_H = self.model(**inputs_H, output_attentions=True, output_hidden_states=True)
        
        inputs_Hp = self.tokenizer(text_Hp, return_tensors='pt', add_special_tokens=False).to(self.device)
        outputs_Hp = self.model(**inputs_Hp, output_attentions=True, output_hidden_states=True)

        # 2. Identify the query token (the last token of the prompt)
        query_idx_H = inputs_H.input_ids.shape[1] - 1
        query_idx_Hp = inputs_Hp.input_ids.shape[1] - 1

        # 3. Sum attention from the query token to the answer span across all specified layers and heads
        total_attention_H = 0.0
        total_attention_Hp = 0.0

        for layer_idx in self.layers:
            # Attention shape: [batch, heads, seq_len, seq_len]
            attentions_on_span_H = outputs_H.attentions[layer_idx][0, :, query_idx_H, indices_H].sum()
            total_attention_H += attentions_on_span_H.item()

            attentions_on_span_Hp = outputs_Hp.attentions[layer_idx][0, :, query_idx_Hp, indices_Hp].sum()
            total_attention_Hp += attentions_on_span_Hp.item()

        # import pdb; pdb.set_trace()  # NEW: Debugging breakpoint
        
        # 4. Check if the increase meets the threshold
        if total_attention_H == 0: # Avoid division by zero
            is_valid = total_attention_Hp > 0
        else:
            ratio = total_attention_Hp / total_attention_H
            is_valid = ratio >= self.attention_threshold

        return is_valid, outputs_H, outputs_Hp


    def run(self, output_dir):
        # 1) buffers per layer, per head
        num_layers = len(self.layers)
        n_kv = self.model.config.num_key_value_heads if "gemma3" not in self.model.__class__.__name__.lower() else self.model.config.text_config.num_key_value_heads
        buf_H = [[[] for _ in range(n_kv)] for _ in range(num_layers)]
        buf_Hp = [[[] for _ in range(n_kv)] for _ in range(num_layers)]

        pbar = tqdm(total=self.max_samples, desc="Extracting Keys", unit="ex")
        count = 0
        samples_kept = 0
        samples_discarded = 0

        for ex in self.iter_examples():
            for ctx, rel_q, ans in self.get_triplets(ex):
                # assemble texts
                text_H, text_Hp = self.assemble_texts(ctx, rel_q)

                # find answer token indices
                idx_H = self.span_token_indices(self.tokenizer, text_H, ans)
                idx_Hp = self.span_token_indices(self.tokenizer, text_Hp, ans)

                if not (idx_H and idx_Hp):
                    continue
                
                # --- NEW: ATTENTION VALIDATION STEP ---
                is_valid, outputs_H, outputs_Hp = self._check_attention_increase(text_H, text_Hp, idx_H, idx_Hp)

                if not is_valid:
                    samples_discarded += 1
                    pbar.set_postfix({"kept": samples_kept, "discarded": samples_discarded})
                    continue
                
                samples_kept += 1
                pbar.set_postfix({"kept": samples_kept, "discarded": samples_discarded})
                # --- END OF VALIDATION ---

                assert len(idx_H) == len(idx_Hp), f"Indices mismatch: {len(idx_H)}, {len(idx_Hp)}"

                # extract keys, reusing the model outputs from the attention check for efficiency
                keys_H = self.extract_keys(self.model, self.tokenizer, text_H, idx_H, self.layers, self.feature, outputs=outputs_H)
                keys_Hp = self.extract_keys(self.model, self.tokenizer, text_Hp, idx_Hp, self.layers, self.feature, outputs=outputs_Hp)

                # distribute into buffers
                for idx_flat, (k_H, k_Hp) in enumerate(zip(keys_H, keys_Hp)):
                    L = idx_flat // n_kv
                    h = idx_flat % n_kv
                    buf_H[L][h].append(k_H)
                    buf_Hp[L][h].append(k_Hp)

            count += 1
            pbar.update(1)

            if count >= self.max_samples:
                break
        pbar.close()
        print(f"Finished key extraction. Kept {samples_kept} samples, discarded {samples_discarded}.")

        # 2) Determine what to compute and save
        if self.save_svd:
            self._compute_and_save_svd(buf_H, buf_Hp, num_layers, n_kv, output_dir)

        if self.save_traditional:
            self._compute_and_save_traditional(buf_H, buf_Hp, num_layers, n_kv, output_dir)

    # ... (the rest of the class, _compute_and_save_svd, _compute_and_save_traditional, span_token_indices, remain the same) ...
    
    @staticmethod
    def extract_keys(model, tokenizer, text: str, indices: list[int], layers: list[int], feature: str, outputs=None) -> list[torch.Tensor]:
        """MODIFIED: Can accept pre-computed model outputs to avoid redundant forward passes."""
        if outputs is None:
            inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).to(model.device)
            outputs = model(**inputs, use_cache=False, output_hidden_states=True)
        
        hiddens = outputs.hidden_states
        result: list[torch.Tensor] = []
        for L in layers:
            h_in = hiddens[L]
            attn = model.model.layers[L].self_attn if "gemma3" not in model.__class__.__name__.lower() else \
            model.language_model.model.layers[L].self_attn

            if "qwen3" in model.__class__.__name__.lower():
                h_in = model.model.layers[L].input_layernorm(h_in)
                if hasattr(attn, 'k_norm'):
                    input_shape = h_in.shape[:-1]
                    dim_h = model.config.head_dim
                    k = attn.k_norm(attn.k_proj(h_in).view(*input_shape, -1, dim_h))[0]
            elif "gemma" in model.__class__.__name__.lower():
                h_in = model.language_model.model.layers[L].input_layernorm(h_in)
                if hasattr(attn, 'k_norm'):
                    input_shape = h_in.shape[:-1]
                    dim_h = model.config.text_config.head_dim
                    k = attn.k_norm(attn.k_proj(h_in).view(*input_shape, -1, dim_h))[0]
            elif "llama" in model.__class__.__name__.lower():
                h_in = model.model.layers[L].input_layernorm(h_in)
                input_shape = h_in.shape[:-1]
                hidden_shape = (*input_shape, -1, model.config.head_dim)
                k = attn.k_proj(h_in).view(hidden_shape)[0]
            elif "mistral" in model.model.layers[L].__class__.__name__.lower():
                h_in = model.model.layers[L].input_layernorm(h_in)
                input_shape = h_in.shape[:-1]
                hidden_shape = (*input_shape, -1, model.config.head_dim)
                k = attn.k_proj(h_in).view(hidden_shape)[0]
            else:
                raise NotImplementedError(f"Unsupported model type: {model.__class__.__name__}.")

            if torch.isnan(k).any():
                raise ValueError("k contains NaN values")

            k = phi(k.float(), feature).to(torch.float)
            k_sel = k[indices]

            for h in range(k_sel.size(1)):
                result.append(k_sel[:, h, :])
        assert len(result) > 0
        return result

    def _compute_and_save_svd(self, buf_H, buf_Hp, num_layers, n_kv, output_dir):
        """Compute and save SVD components (U matrices and singular values)"""
        pos_U_list, pos_S_list = [], []
        applied, skipped = [], []

        for L in tqdm(range(num_layers), desc="Computing SVD Components", unit="layer"):
            Up_heads, Sp_heads = [], []

            for h in range(n_kv):
                H_mat = torch.cat(buf_H[L][h], 0).double().to(self.device)
                Hp_mat = torch.cat(buf_Hp[L][h], 0).double().to(self.device)

                # cross-cov → SVD
                Omega_p = (H_mat.T @ Hp_mat) / H_mat.size(0)

                try:
                    Up, Sp, _ = torch.linalg.svd(Omega_p.float(), full_matrices=False)
                except Exception as e:
                    print(f"SVD failed for layer {L}, head {h}: {e}")
                    # Fallback for problematic matrices
                    d = Omega_p.shape[0]
                    Up = torch.eye(d, device=self.device, dtype=torch.float)
                    Sp = torch.ones(d, device=self.device, dtype=torch.float)

                norm_value = (torch.norm(Hp_mat - H_mat) / len(Hp_mat)).item()

                # decide whether to apply or skip
                if norm_value < self.min_diff:
                    skipped.append((self.layers[L], h, norm_value))
                    # Mark as invalid with NaN - will be filtered out during coefficient computation
                    Up = torch.full_like(Up, float('nan'))
                    Sp = torch.full_like(Sp, float('nan'))
                else:
                    applied.append((self.layers[L], h, norm_value))

                Up_heads.append(Up.to(torch.float))
                Sp_heads.append(Sp.to(torch.float))

            pos_U_list.append(torch.stack(Up_heads, dim=0))  # (H, d, d)
            pos_S_list.append(torch.stack(Sp_heads, dim=0))  # (H, d)

        # stack layers → (num_layers, H, d, d) and (num_layers, H, d)
        pos_U = torch.stack(pos_U_list, dim=0)
        pos_S = torch.stack(pos_S_list, dim=0)

        # save SVD components
        os.makedirs(output_dir, exist_ok=True)
        model_name = self.model_path.split('/')[-1]

        # Save positive SVD components
        pos_svd_data = {
            'layers': self.layers,
            'U_matrices': pos_U.cpu(),
            'singular_values': pos_S.cpu()
        }
        pos_filename = f"{model_name}_{self.min_diff}mindiff_pos_svd.pt"
        if self.feature:
            pos_filename = f"{model_name}_{self.min_diff}mindiff_pos_svd_{self.feature}.pt"
        torch.save(pos_svd_data, os.path.join(output_dir, pos_filename))

        # summary
        print(f"\nSVD Components Summary:")
        if applied:
            print(f" ✔ Applied SVD: {len(applied)}")
            print(f"{applied}")
        if skipped:
            print(f" ✖ Skipped (zero): {len(skipped)}")
            print(f"{skipped}")

        print(f"Saved positive SVD to {output_dir}, U: {tuple(pos_U.shape)}, S: {tuple(pos_S.shape)}")
        print(f"Files: {pos_filename}")

    def _compute_and_save_traditional(self, buf_H, buf_Hp, num_layers, n_kv, output_dir):
        """Compute and save traditional projection matrices"""
        pos_list = []
        applied, skipped = [], []
        norm_diffs = np.zeros((num_layers, n_kv))

        for L in tqdm(range(num_layers), desc="Computing Traditional Projectors", unit="layer"):
            Pp_heads = []
            for h in range(n_kv):
                H_mat = torch.cat(buf_H[L][h], 0).double().to(self.device)
                Hp_mat = torch.cat(buf_Hp[L][h], 0).double().to(self.device)

                # cross-cov → Pp, Pn
                Omega_p = (H_mat.T @ Hp_mat) / H_mat.size(0)

                Up, Sp, _ = torch.linalg.svd(Omega_p.float(), full_matrices=False)
                kp = (Sp.cumsum(0) / Sp.sum() < self.top_pct).sum().item() + 1
                Pp = (Up[:, :kp] @ Up[:, :kp].T).to(torch.float)

                norm_value = (torch.norm(Hp_mat - H_mat) / len(Hp_mat)).item()
                norm_diffs[L, h] = norm_value

                # decide
                if norm_value < self.min_diff:
                    skipped.append((self.layers[L], h, norm_value))
                    # Mark as invalid with NaN
                    Pp = torch.full_like(Pp, float('nan'))
                else:
                    applied.append((self.layers[L], h, norm_value))

                Pp_heads.append(Pp)

            pos_list.append(torch.stack(Pp_heads, dim=0))  # (H,d,d)

        # stack layers → (num_layers, H, d, d)
        pos_proj = torch.stack(pos_list, dim=0)

        # save
        os.makedirs(output_dir, exist_ok=True)
        model_name = self.model_path.split('/')[-1]

        torch.save({'layers': self.layers, 'proj': pos_proj.cpu()}, os.path.join(output_dir,
                                                                                 f"{model_name}_pos_proj_{self.feature}.pt") if self.feature else os.path.join(
            output_dir, f"{model_name}_pos_proj.pt"))

        # summary
        print(f"\nTraditional Projection Summary:")
        if applied:
            print(f" ✔ Applied projection: {applied}")
        if skipped:
            print(f" ✖ Skipped (identity): {skipped}")

        print(f"Saved positive projectors to {output_dir}, {tuple(pos_proj.shape)}")

    @staticmethod
    def span_token_indices(tokenizer, text: str, sub: str) -> list[int] | None:
        low, sub_low = text.lower(), sub.lower()
        if sub_low not in low:
            return None
        start = low.index(sub_low)
        end = start + len(sub_low)
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        span_indices = [i for i, (s, e) in enumerate(enc.offset_mapping) if s >= start and e <= end]
        if len(span_indices) == 0:
            span_indices = [i for i, (s, e) in enumerate(enc.offset_mapping) if s >= (start - 1) and e <= end]
        if len(span_indices) == 0:
            span_indices = [i for i, (s, e) in enumerate(enc.offset_mapping) if s >= start and e <= (end + 1)]

        return span_indices

