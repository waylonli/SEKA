# src/model/seka_llm_adaptive.py
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from src.utils import encode_with_markers, _parse_layers, _load_proj, phi, phi_inv


class TaskSpecificProjector:
    """
    Training-free task-specific projector that dynamically combines expert projections
    based on query-singular vector alignment.
    """

    def __init__(self, expert_svd_data: dict, device: str = "cuda", multi_gpu: bool = False):
        """
        Args:
            expert_svd_data: Dict mapping expert names to (layers_info, U_matrices, singular_values)
            device: Device to place tensors on
            multi_gpu: Whether model is using multi-GPU setup
        """
        self.device = device
        self.multi_gpu = multi_gpu
        self.expert_names = list(expert_svd_data.keys())
        self.num_experts = len(self.expert_names)

        # Store U matrices and singular values directly
        self.expert_U_matrices = {}
        self.expert_singular_values = {}

        for expert_name, (layers_info, U_matrices, singular_values) in expert_svd_data.items():
            # Keep matrices on CPU for multi-GPU to avoid memory issues, move to device as needed
            if multi_gpu:
                self.expert_U_matrices[expert_name] = U_matrices.cpu()
                self.expert_singular_values[expert_name] = singular_values.cpu()
            else:
                self.expert_U_matrices[expert_name] = U_matrices.to(device)
                self.expert_singular_values[expert_name] = singular_values.to(device)

        # Get dimensions from first expert
        first_expert = self.expert_names[0]
        L, H, d, _ = self.expert_U_matrices[first_expert].shape
        self.num_layers, self.num_heads, self.head_dim = L, H, d

    def get_layer_device(self, query_device: torch.device) -> torch.device:
        """Get the appropriate device for matrix operations, considering multi-GPU setup."""
        if self.multi_gpu:
            # Use the same device as the query tensor for consistency
            return query_device
        else:
            return self.device

    def compute_dynamic_coefficients(self, query: torch.Tensor, layer_idx: int, head_idx: int,
                                     top_k: int = None,
                                     combination_method: str = "weighted_top_k") -> torch.Tensor:
        """
        Compute dynamic coefficients for expert combination based on query alignment.

        Args:
            query: Query tensor (d,) - normalized query from last token
            layer_idx: Layer index in selected layers
            head_idx: Head index
            top_k: Number of top singular vectors to use (if None, use all)
            combination_method: How to combine singular vectors and values
                - "weighted_top_k": Use top-k singular vectors weighted by singular values
                - "all_weighted": Use all singular vectors weighted by singular values
                - "top_k_uniform": Use top-k singular vectors with uniform weighting

        Returns:
            Dynamic coefficients for each expert (num_experts,)
        """
        if query.dim() == 2:
            query = query.squeeze(0)  # (d,)

        # Normalize query
        query_norm = query / (torch.norm(query) + 1e-8)
        target_device = self.get_layer_device(query.device)

        coefficients = []

        for expert_name in self.expert_names:
            # Get U matrix and singular values for this expert, layer, and head
            U = self.expert_U_matrices[expert_name][layer_idx, head_idx]  # (d, d)
            S = self.expert_singular_values[expert_name][layer_idx, head_idx]  # (d,)
            
            # Move to target device if needed (for multi-GPU)
            if U.device != target_device:
                U = U.to(target_device)
                S = S.to(target_device)

            # Skip invalid/NaN matrices (heads that didn't meet min_diff threshold)
            if torch.isnan(U).any() or torch.isnan(S).any():
                coefficient = torch.tensor(float('-inf'), device=query.device)  # Will be near 0 after softmax
                coefficients.append(coefficient)
                continue

            # Determine how many components to use
            if top_k is None:
                k = U.shape[1]
            else:
                k = min(top_k, U.shape[1])

            if combination_method == "weighted_top_k":
                # Use top-k singular vectors weighted by their singular values
                top_U = U[:, :k]  # (d, k) - top k singular vectors
                top_S = S[:k]  # (k,) - top k singular values

                # Compute alignment: q^T * U * S
                alignments = torch.matmul(query_norm, top_U)  # (k,)
                weighted_alignments = alignments * top_S  # (k,)
                coefficient = torch.sum(weighted_alignments)  # scalar

            elif combination_method == "all_weighted":
                # Use all singular vectors weighted by singular values
                alignments = torch.matmul(query_norm, U)  # (d,)
                weighted_alignments = alignments * S  # (d,)
                coefficient = torch.sum(weighted_alignments)  # scalar

            elif combination_method == "top_k_uniform":
                # Use top-k singular vectors with uniform weighting
                top_U = U[:, :k]  # (d, k)
                alignments = torch.matmul(query_norm, top_U)  # (k,)
                coefficient = torch.mean(alignments)  # scalar

            else:
                raise ValueError(f"Unknown combination method: {combination_method}")

            coefficients.append(coefficient)

        coefficients = torch.stack(coefficients)  # (num_experts,)
        
        # Use softmax for probabilistic mixing instead of L2 normalization
        coefficients = torch.softmax(coefficients, dim=0)

        # Debug coefficient values
        if torch.isnan(coefficients).any() or torch.isinf(coefficients).any():
            # print(f"Warning: Invalid coefficients detected at layer {layer_idx}, head {head_idx}")
            # print(f"Raw coefficients before softmax: {torch.stack([torch.sum(c) for c in coefficients])}")
            coefficients = torch.ones(self.num_experts, device=coefficients.device) / self.num_experts

        return coefficients

    def get_dynamic_projection(self, query: torch.Tensor, layer_idx: int, head_idx: int,
                               top_k: int = 3, combination_method: str = "weighted_top_k") -> torch.Tensor:
        """
        Get query-adaptive projection matrix: P_dyn = sum(α_m * P_m)
        where P_m = U_m @ U_m^T (reconstruct projection from stored U matrices)

        Args:
            query: Query tensor (d,)
            layer_idx: Layer index in the selected layers
            head_idx: Head index
            top_k: Number of top singular vectors to use
            combination_method: How to combine singular vectors and values

        Returns:
            Dynamic projection matrix (d, d)
        """
        coefficients = self.compute_dynamic_coefficients(
            query, layer_idx, head_idx, top_k, combination_method
        )

        target_device = self.get_layer_device(query.device)
        dynamic_proj = torch.zeros(self.head_dim, self.head_dim, device=target_device, dtype=torch.float32)

        for i, expert_name in enumerate(self.expert_names):
            U = self.expert_U_matrices[expert_name][layer_idx, head_idx]  # (d, d)
            
            # Move to target device if needed (for multi-GPU)
            if U.device != target_device:
                U = U.to(target_device)
            
            # Skip invalid/NaN matrices 
            if torch.isnan(U).any():
                continue

            if top_k is not None:
                k = min(top_k, U.shape[1])
                U_truncated = U[:, :k]  # (d, k)
                expert_proj = torch.matmul(U_truncated, U_truncated.T)  # (d, d)
            else:
                expert_proj = torch.matmul(U, U.T)  # (d, d)

            dynamic_proj += coefficients[i] * expert_proj

        return dynamic_proj


class AdaptiveSEKALLM:
    """Enhanced SEKA with query-adaptive projections using training-free approach"""

    def __init__(self,
                 model_or_path: str,
                 *,
                 device: str | None = "auto",
                 marker_start: str = "**",
                 marker_end: str | None = None,
                 expert_paths: dict = None,
                 layers: str = "last10",
                 amplify_factor: float = 1.0,
                 feature_function: str | None = None,
                 top_k_singular: int = 3,
                 combination_method: str = "weighted_top_k",
                 **hf_kwargs
                 ):
        if device == "auto":
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
            else "cpu")

        multi_gpu = torch.cuda.device_count() > 1 and str(device).startswith("cuda")

        self.name_or_path = f"AdaptiveSEKA-{model_or_path}"
        self.tok: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_or_path, padding_side="left", **hf_kwargs)

        if multi_gpu:
            print(f"Initializing AdaptiveSEKA with {torch.cuda.device_count()} GPUs")
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model_or_path, device_map="auto", **hf_kwargs).eval()
            self.multi_gpu = True
            # For multi-GPU, use the device of the first layer for expert matrices  
            self.expert_device = next(self.model.parameters()).device
            print(f"Expert matrices will be managed with base device: {self.expert_device}")
        else:
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model_or_path, **hf_kwargs).to(device).eval()
            self.multi_gpu = False
            self.expert_device = device

        self.m_start, self.m_end = marker_start, (marker_start if marker_end is None else marker_end)
        self.layers = layers
        self.amplify_factor = amplify_factor
        self.feature_function = feature_function
        self.top_k_singular = top_k_singular
        self.combination_method = combination_method

        if expert_paths is None:
            raise ValueError("expert_paths must be provided for adaptive SEKA")

        expert_svd_data = {}
        for expert_name, expert_path in expert_paths.items():
            layers_info, U_matrices, singular_values = self._load_svd_data(expert_path, self.expert_device)
            expert_svd_data[expert_name] = (layers_info, U_matrices, singular_values)

        self.task_projector = TaskSpecificProjector(expert_svd_data, self.expert_device, multi_gpu=self.multi_gpu)
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        n_layers = len(self.model.model.layers) if "gemma3" not in self.model.__class__.__name__.lower() else len(
            self.model.language_model.model.layers)
        self.sel_layers = _parse_layers(layers, n_layers)

    @property
    def device(self):
        if self.multi_gpu:
            # For multi-GPU, return the device of the first parameter
            return next(self.model.parameters()).device
        else:
            return self.model.device
    
    def get_layer_device(self, layer_idx: int) -> torch.device:
        """Get the device where a specific layer is located."""
        if self.multi_gpu:
            if "gemma3" not in self.model.__class__.__name__.lower():
                layer = self.model.model.layers[layer_idx]
            else:
                layer = self.model.language_model.model.layers[layer_idx]
            return next(layer.parameters()).device
        else:
            return self.device

    def _load_svd_data(self, path: str, device):
        obj = torch.load(path, map_location=device)
        if isinstance(obj, dict):
            layers = obj.get('layers', None)
            if 'U_matrices' in obj and 'singular_values' in obj:
                U_matrices = obj['U_matrices'].to(device)
                singular_values = obj['singular_values'].to(device)
            else:
                proj = obj['proj'].to(device)
                U_matrices, singular_values = self._decompose_projections(proj)
        else:
            layers = None
            proj = obj.to(device)
            U_matrices, singular_values = self._decompose_projections(proj)

        if U_matrices.ndim == 2:
            U_matrices = U_matrices.unsqueeze(0).unsqueeze(0)
            singular_values = singular_values.unsqueeze(0).unsqueeze(0)
        elif U_matrices.ndim == 3:
            U_matrices = U_matrices.unsqueeze(1)
            singular_values = singular_values.unsqueeze(1)
        
        return layers, U_matrices, singular_values

    def _decompose_projections(self, projections: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        L, H, d, _ = projections.shape
        U_matrices = torch.zeros(L, H, d, d, device=projections.device)
        singular_values = torch.zeros(L, H, d, device=projections.device)
        for l in range(L):
            for h in range(H):
                proj = projections[l, h]
                try:
                    U, S, V = torch.svd(proj)
                    U_matrices[l, h] = U
                    singular_values[l, h] = S
                except Exception as e:
                    print(f"SVD decomposition failed for layer {l}, head {h}: {e}")
                    U_matrices[l, h] = torch.eye(d, device=projections.device)
                    singular_values[l, h] = torch.ones(d, device=projections.device)
        return U_matrices, singular_values

    def generate(self,
                 ids: torch.LongTensor | str,
                 steer: bool = True,
                 steer_mask: torch.Tensor | None = None,
                 attention_mask: torch.Tensor | None = None,
                 return_raw: bool = False,
                 **gen_kw) -> str:

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

        if steer:
            self.attach_adaptive_projection(input_ids=ids, steer_mask_tensor=steer_mask, silence=True)
        else:
            self.remove_projection()

        if "attention_mask" not in gen_kw and attention_mask is not None:
            gen_kw["attention_mask"] = attention_mask

        out = self.model.generate(ids, **gen_kw)

        if steer:
            self.remove_projection()

        if return_raw:
            return out

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(ids, out)
        ]
        generated = self.tok.batch_decode(generated_ids, skip_special_tokens=True)

        return generated[0] if len(generated) == 1 else generated

    def _get_queries_from_steered_tokens(self, hidden_states: torch.Tensor, layer_idx: int, 
                                        steer_mask: torch.Tensor) -> torch.Tensor:
        """Extract queries from tokens that will be steered, not just last token."""
        # Get indices of tokens that will be steered
        steer_indices = torch.where(steer_mask[0])[0]  # Assuming batch size 1
        
        if len(steer_indices) == 0:
            # Fallback to last token if no steered tokens
            steer_indices = torch.tensor([hidden_states.shape[1] - 1], device=hidden_states.device)
        
        # Use the first steered token for routing (could also average multiple)
        token_hidden = hidden_states[0, steer_indices[0], :]

        if "gemma3" not in self.model.__class__.__name__.lower():
            attn_layer = self.model.model.layers[layer_idx].self_attn
            layernorm = self.model.model.layers[layer_idx].input_layernorm
        else:
            attn_layer = self.model.language_model.model.layers[layer_idx].self_attn
            layernorm = self.model.language_model.model.layers[layer_idx].input_layernorm
        
        normalized_hidden = layernorm(token_hidden.unsqueeze(0).unsqueeze(0))[0, 0, :]
        q_proj = attn_layer.q_proj(normalized_hidden)
        
        if "gemma3" in self.model.__class__.__name__.lower():
            num_q_heads = self.model.config.text_config.num_attention_heads
            head_dim = self.model.config.text_config.head_dim
        else:
            num_q_heads = self.model.config.num_attention_heads
            head_dim = self.model.config.head_dim

        q = q_proj.view(num_q_heads, head_dim)

        num_kv_heads = self.task_projector.num_heads
        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0, f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            num_groups = num_q_heads // num_kv_heads
            q = q.view(num_kv_heads, num_groups, head_dim).mean(dim=1)

        return q

    def attach_adaptive_projection(self,
                                   input_ids: torch.Tensor,
                                   steer_mask_tensor=None,
                                   amplify_factor=None,
                                   silence=False):
        self.remove_projection()
        amplify_factor = self.amplify_factor if amplify_factor is None else amplify_factor
        
        first_dev = self.device
        # For multi-GPU, we'll handle device placement per layer
        if steer_mask_tensor is not None:
            m_dev = steer_mask_tensor.unsqueeze(0) if steer_mask_tensor.dim() == 1 else steer_mask_tensor
        else:
            m_dev = None
        
        root = self.model.module if hasattr(self.model, "module") else self.model
        
        # --- OPTIMIZATION: Single forward pass and pre-computation ---
        layer_dynamic_projections = {}
        if input_ids is not None:
            # 1. Perform a single forward pass to get all hidden states
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
            
            # 2. Iterate through layers to compute and store dynamic projections
            for i, actual_layer in enumerate(self.sel_layers):
                hidden_states = outputs.hidden_states[actual_layer + 1]
                try:
                    queries = self._get_queries_from_steered_tokens(hidden_states, actual_layer, steer_mask_tensor)
                    
                    for h in range(self.task_projector.num_heads):
                        query_for_head = queries[h, :]
                        dynamic_proj = self.task_projector.get_dynamic_projection(
                            query_for_head, i, h, self.top_k_singular, self.combination_method
                        )
                        # Store projection on the same device as the layer it will be applied to
                        if self.multi_gpu:
                            layer_device = self.get_layer_device(actual_layer)
                            layer_dynamic_projections[(i, h)] = dynamic_proj.to(layer_device)
                        else:
                            layer_dynamic_projections[(i, h)] = dynamic_proj

                except Exception as e:
                    print(f"Warning: Failed to pre-compute projections for layer {actual_layer}: {e}")

        for i, L in enumerate(self.sel_layers):
            attn = root.model.layers[L].self_attn if "gemma3" not in root.__class__.__name__.lower() else root.language_model.model.layers[L].self_attn
            mod = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj
            
            layer_device = next(mod.parameters()).device
            m_dev_layer = m_dev.to(layer_device) if m_dev is not None else None

            def _adaptive_hook(module, input_tensor, k_in,
                               m=m_dev_layer,
                               layer_idx=i,
                               amp_factor=amplify_factor,
                               precomputed_projs=layer_dynamic_projections):

                if k_in.dim() == 4: B, T, H, D = k_in.shape; k_view = k_in
                elif k_in.dim() == 3: B, T, D_all = k_in.shape; H, D = self.task_projector.num_heads, self.task_projector.head_dim; k_view = k_in.view(B, T, H, D)
                else: raise ValueError(f"Unsupported k_in shape: {k_in.shape}")

                if m is None or m.sum() == 0: return k_in

                k_feat = phi(k_view, self.feature_function)
                
                for h in range(H):
                    if m.shape[1] != T: continue
                    
                    # --- OPTIMIZATION: Use pre-computed projection ---
                    dynamic_proj = precomputed_projs.get((layer_idx, h))
                    if dynamic_proj is None:
                        continue # Skip if pre-computation failed

                    # Ensure projection is on the same device as key features
                    if dynamic_proj.device != k_feat.device:
                        dynamic_proj = dynamic_proj.to(k_feat.device)

                    for b in range(B):
                        mask_b = m[b]
                        if mask_b.sum() == 0: continue
                        
                        k_sel = k_feat[b, mask_b, h, :]
                        if k_sel.numel() > 0:
                            # Apply projection: P @ k (not k @ P^T)
                            delta = torch.matmul(dynamic_proj, k_sel.T).T
                            k_feat[b, mask_b, h, :] += amp_factor * delta

                return k_feat.contiguous().view(B, T, H * D) if k_in.dim() == 3 else k_feat

            self._hooks.append(mod.register_forward_hook(_adaptive_hook))

        if not silence:
            if self.multi_gpu:
                print(f"✅ Adaptive steering hooks attached on layers {self.sel_layers} (Multi-GPU)")
                print(f"   Projections distributed across devices for optimal performance")
            else:
                print(f"✅ Adaptive steering hooks attached on layers {self.sel_layers}")

    def remove_projection(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()

    def set_amplify_factor(self, factor: float): self.amplify_factor = factor
    def set_combination_method(self, method: str): self.combination_method = method
    def set_top_k_singular(self, k: int): self.top_k_singular = k
    
    def debug_coefficient_statistics(self, input_ids: torch.Tensor, steer_mask_tensor: torch.Tensor):
        """Print coefficient statistics for debugging routing behavior."""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
            
            print("=== Coefficient Statistics ===")
            if self.multi_gpu:
                print("Multi-GPU setup detected - device placement:")
            
            for i, actual_layer in enumerate(self.sel_layers):
                hidden_states = outputs.hidden_states[actual_layer + 1]
                if self.multi_gpu:
                    layer_device = self.get_layer_device(actual_layer)
                    print(f"Layer {actual_layer} on device: {layer_device}")
                
                queries = self._get_queries_from_steered_tokens(hidden_states, actual_layer, steer_mask_tensor)
                
                for h in range(self.task_projector.num_heads):
                    query_for_head = queries[h, :]
                    coefficients = self.task_projector.compute_dynamic_coefficients(
                        query_for_head, i, h, self.top_k_singular, self.combination_method
                    )
                    print(f"Layer {actual_layer}, Head {h}: {[f'{self.task_projector.expert_names[j]}={coefficients[j]:.3f}' for j in range(len(coefficients))]}")
            print("==========================")
