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

    def __init__(self, expert_svd_data: dict, device: str = "cuda"):
        """
        Args:
            expert_svd_data: Dict mapping expert names to (layers_info, U_matrices, singular_values)
            device: Device to place tensors on
        """
        self.device = device
        self.expert_names = list(expert_svd_data.keys())
        self.num_experts = len(self.expert_names)

        # Store U matrices and singular values directly
        self.expert_U_matrices = {}
        self.expert_singular_values = {}

        for expert_name, (layers_info, U_matrices, singular_values) in expert_svd_data.items():
            self.expert_U_matrices[expert_name] = U_matrices.to(device)
            self.expert_singular_values[expert_name] = singular_values.to(device)

        # Get dimensions from first expert
        first_expert = self.expert_names[0]
        L, H, d, _ = self.expert_U_matrices[first_expert].shape
        self.num_layers, self.num_heads, self.head_dim = L, H, d

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

        coefficients = []

        for expert_name in self.expert_names:
            # Get U matrix and singular values for this expert, layer, and head
            U = self.expert_U_matrices[expert_name][layer_idx, head_idx]  # (d, d)
            S = self.expert_singular_values[expert_name][layer_idx, head_idx]  # (d,)

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

        # Optional normalization (can be modified later)
        # Method 1: Softmax normalization
        # coefficients = torch.softmax(coefficients / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)), dim=0)

        # Method 2: L2 normalization (alternative - can be switched)
        coefficients = coefficients / (torch.norm(coefficients) + 1e-8)

        # Method 3: Raw coefficients (alternative - can be switched)
        # coefficients = coefficients

        # if coeffcients are not all 0, check
        if not torch.all(coefficients == 0):
            print(f"[AdaptiveSEKA] Coefficients for layer {layer_idx}, head {head_idx}: {coefficients.cpu().numpy()}")
            import pdb; pdb.set_trace()

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

        # Combine expert projections: P_dyn = sum(α_m * U_m @ U_m^T)
        dynamic_proj = torch.zeros(self.head_dim, self.head_dim, device=self.device, dtype=torch.float32)

        for i, expert_name in enumerate(self.expert_names):
            # Reconstruct projection matrix from U: P = U @ U^T
            U = self.expert_U_matrices[expert_name][layer_idx, head_idx]  # (d, d)

            # Decide how many components to use for projection reconstruction
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
                 expert_paths: dict = None,  # {'fact': 'path/to/fact_svd.pt', 'instruction': 'path/to/inst_svd.pt'}
                 layers: str = "last10",
                 amplify_factor: float = 1.0,  # Single amplification factor for dynamic projections
                 feature_function: str | None = None,
                 top_k_singular: int = 3,  # Number of top singular vectors to use
                 combination_method: str = "weighted_top_k",  # How to combine singular vectors/values
                 **hf_kwargs
                 ):
        # Device selection
        if device == "auto":
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
            else "cpu")

        multi_gpu = torch.cuda.device_count() > 1 and str(device).startswith("cuda")

        # HF objects
        self.name_or_path = f"AdaptiveSEKA-{model_or_path}"
        self.tok: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_or_path, padding_side="left", **hf_kwargs)

        if multi_gpu:
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model_or_path,
                device_map="auto",
                **hf_kwargs
            ).eval()
        else:
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model_or_path,
                **hf_kwargs
            ).to(device).eval()

        # Configuration
        self.m_start, self.m_end = marker_start, (marker_start if marker_end is None else marker_end)
        self.layers = layers
        self.amplify_factor = amplify_factor  # Single amplification factor
        self.feature_function = feature_function
        self.top_k_singular = top_k_singular
        self.combination_method = combination_method

        # Load expert SVD data (U matrices and singular values)
        if expert_paths is None:
            raise ValueError("expert_paths must be provided for adaptive SEKA")

        expert_svd_data = {}
        for expert_name, expert_path in expert_paths.items():
            layers_info, U_matrices, singular_values = self._load_svd_data(expert_path, device)
            expert_svd_data[expert_name] = (layers_info, U_matrices, singular_values)

        # Initialize task-specific projector
        self.task_projector = TaskSpecificProjector(expert_svd_data, device)

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        # Parse layers
        n_layers = len(self.model.model.layers) if "gemma3" not in self.model.__class__.__name__.lower() else len(
            self.model.language_model.model.layers)
        self.sel_layers = _parse_layers(layers, n_layers)

    @property
    def device(self):
        return self.model.device

        # Expose everything from the HF model
        object.__setattr__(self, "__getattr__", lambda n: getattr(self.model, n))

    def _load_svd_data(self, path: str, device):
        """
        Load U matrices and singular values from saved projection files.
        Expects files to contain either direct SVD data or projection matrices to decompose.

        Args:
            path: Path to the saved projection file
            device: Device to load tensors on

        Returns:
            layers_info: Layer information
            U_matrices: (L, H, d, d) tensor of U matrices
            singular_values: (L, H, d) tensor of singular values
        """
        obj = torch.load(path, map_location=device)

        if isinstance(obj, dict):
            layers = obj.get('layers', None)

            # Check if file already contains SVD data
            if 'U_matrices' in obj and 'singular_values' in obj:
                U_matrices = obj['U_matrices'].to(device)
                singular_values = obj['singular_values'].to(device)
            else:
                # Decompose projections to get U matrices and singular values
                proj = obj['proj'].to(device)
                U_matrices, singular_values = self._decompose_projections(proj)
        else:
            layers = None
            proj = obj.to(device)
            U_matrices, singular_values = self._decompose_projections(proj)

        # Ensure 4D format for U matrices
        if U_matrices.ndim == 2:
            d = U_matrices.size(0)
            U_matrices = U_matrices.unsqueeze(0).unsqueeze(0)
            singular_values = singular_values.unsqueeze(0).unsqueeze(0)
        elif U_matrices.ndim == 3:
            U_matrices = U_matrices.unsqueeze(1)
            singular_values = singular_values.unsqueeze(1)
        elif U_matrices.ndim == 4:
            pass
        else:
            raise ValueError(f"Unsupported U_matrices dimension: {U_matrices.ndim}")

        return layers, U_matrices, singular_values

    def _decompose_projections(self, projections: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose projection matrices into U matrices and singular values.

        Args:
            projections: Projection tensor (L, H, d, d)

        Returns:
            U_matrices: (L, H, d, d) tensor of U matrices
            singular_values: (L, H, d) tensor of singular values
        """
        L, H, d, _ = projections.shape
        U_matrices = torch.zeros(L, H, d, d, device=projections.device)
        singular_values = torch.zeros(L, H, d, device=projections.device)

        for l in range(L):
            for h in range(H):
                proj = projections[l, h]  # (d, d)

                try:
                    # Compute SVD: proj = U @ diag(S) @ V^T
                    U, S, V = torch.svd(proj)
                    U_matrices[l, h] = U  # (d, d)
                    singular_values[l, h] = S  # (d,)
                except Exception as e:
                    print(f"SVD decomposition failed for layer {l}, head {h}: {e}")
                    # Fallback: identity matrix and uniform values
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

        # Optional steering
        if steer:
            self.attach_adaptive_projection(steer_mask_tensor=steer_mask, silence=True)
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

    def _extract_queries_from_context(self, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Extract per-head query embeddings from the last token that match key dimensions.

        Args:
            input_ids: Input token IDs (B, T)
            layer_idx: Actual layer index in the model (not selected layers index)

        Returns:
            Query embeddings for all heads (num_heads, head_dim) - matching key dimensions per head
        """
        with torch.no_grad():
            # Get hidden states up to the specified layer
            inputs = {"input_ids": input_ids, "output_hidden_states": True, "use_cache": False}
            outputs = self.model(**inputs)

            # Extract hidden state from the specified layer
            hidden_states = outputs.hidden_states[layer_idx + 1]  # +1 because outputs include embedding layer

            # Get the last token's hidden state
            last_token_hidden = hidden_states[0, -1, :]  # (d,) - take first batch, last token

            # Get attention layer
            if "gemma3" not in self.model.__class__.__name__.lower():
                attn_layer = self.model.model.layers[layer_idx].self_attn
            else:
                attn_layer = self.model.language_model.model.layers[layer_idx].self_attn

            # Apply input normalization first (same as in key extraction)
            if "qwen3" in self.model.__class__.__name__.lower():
                if "gemma3" not in self.model.__class__.__name__.lower():
                    normalized_hidden = self.model.model.layers[layer_idx].input_layernorm(
                        last_token_hidden.unsqueeze(0).unsqueeze(0))
                else:
                    normalized_hidden = self.model.language_model.model.layers[layer_idx].input_layernorm(
                        last_token_hidden.unsqueeze(0).unsqueeze(0))
                normalized_hidden = normalized_hidden[0, 0, :]  # (d,)
            elif "gemma" in self.model.__class__.__name__.lower():
                normalized_hidden = self.model.language_model.model.layers[layer_idx].input_layernorm(
                    last_token_hidden.unsqueeze(0).unsqueeze(0))
                normalized_hidden = normalized_hidden[0, 0, :]  # (d,)
            elif "llama" in self.model.__class__.__name__.lower():
                normalized_hidden = self.model.model.layers[layer_idx].input_layernorm(
                    last_token_hidden.unsqueeze(0).unsqueeze(0))
                normalized_hidden = normalized_hidden[0, 0, :]  # (d,)
            elif "mistral" in self.model.model.layers[layer_idx].__class__.__name__.lower():
                normalized_hidden = self.model.model.layers[layer_idx].input_layernorm(
                    last_token_hidden.unsqueeze(0).unsqueeze(0))
                normalized_hidden = normalized_hidden[0, 0, :]  # (d,)
            else:
                normalized_hidden = last_token_hidden

            # Extract queries using the SAME process as key extraction
            # This ensures dimensional consistency per head
            if "qwen3" in self.model.__class__.__name__.lower():
                if hasattr(attn_layer, 'q_norm'):
                    input_shape = normalized_hidden.unsqueeze(0).unsqueeze(0).shape[:-1]  # (1, 1)
                    dim_h = self.model.config.head_dim
                    # Project and reshape: (1, 1, hidden) -> (1, 1, num_heads, head_dim)
                    q_proj = attn_layer.q_proj(normalized_hidden.unsqueeze(0).unsqueeze(0))
                    q_reshaped = q_proj.view(*input_shape, -1, dim_h)  # (1, 1, num_heads, head_dim)
                    # Apply q_norm if present
                    q = attn_layer.q_norm(q_reshaped)[0, 0]  # (num_heads, head_dim)
                else:
                    # Fallback: direct projection and reshape
                    q_proj = attn_layer.q_proj(normalized_hidden)  # (num_heads * head_dim,)
                    dim_h = self.model.config.head_dim
                    q = q_proj.view(-1, dim_h)  # (num_heads, head_dim)

            elif "gemma" in self.model.__class__.__name__.lower():
                if hasattr(attn_layer, 'q_norm'):
                    input_shape = normalized_hidden.unsqueeze(0).unsqueeze(0).shape[:-1]
                    dim_h = self.model.config.text_config.head_dim
                    q_proj = attn_layer.q_proj(normalized_hidden.unsqueeze(0).unsqueeze(0))
                    q_reshaped = q_proj.view(*input_shape, -1, dim_h)
                    q = attn_layer.q_norm(q_reshaped)[0, 0]  # (num_heads, head_dim)
                else:
                    q_proj = attn_layer.q_proj(normalized_hidden)
                    dim_h = self.model.config.text_config.head_dim
                    q = q_proj.view(-1, dim_h)

            elif "llama" in self.model.__class__.__name__.lower():
                input_shape = normalized_hidden.unsqueeze(0).unsqueeze(0).shape[:-1]
                hidden_shape = (*input_shape, -1, self.model.config.head_dim)
                q_proj = attn_layer.q_proj(normalized_hidden.unsqueeze(0).unsqueeze(0))
                q = q_proj.view(hidden_shape)[0, 0]  # (num_heads, head_dim)

            elif "mistral" in attn_layer.__class__.__name__.lower():
                input_shape = normalized_hidden.unsqueeze(0).unsqueeze(0).shape[:-1]
                hidden_shape = (*input_shape, -1, self.model.config.head_dim)
                q_proj = attn_layer.q_proj(normalized_hidden.unsqueeze(0).unsqueeze(0))
                q = q_proj.view(hidden_shape)[0, 0]  # (num_heads, head_dim)

            else:
                raise NotImplementedError(f"Unsupported model type: {self.model.__class__.__name__}")

            # Handle MQA/GQA: if we have more query heads than key/value heads
            num_q_heads = q.shape[0]
            num_kv_heads = self.task_projector.num_heads

            if num_q_heads != num_kv_heads:
                if num_q_heads > num_kv_heads:
                    # MQA/GQA case: more query heads than KV heads
                    # We need to group query heads to match KV heads
                    group_size = num_q_heads // num_kv_heads
                    remainder = num_q_heads % num_kv_heads

                    # Group query heads by averaging within each group
                    grouped_queries = []
                    for kv_head in range(num_kv_heads):
                        start_idx = kv_head * group_size
                        end_idx = start_idx + group_size

                        # Handle remainder by distributing extra heads to first few groups
                        if kv_head < remainder:
                            end_idx += 1
                            start_idx += kv_head
                        else:
                            start_idx += remainder
                            end_idx += remainder

                        # Average the query heads in this group
                        if start_idx < num_q_heads:
                            group_queries = q[start_idx:min(end_idx, num_q_heads)]  # (group_size, head_dim)
                            averaged_query = group_queries.mean(dim=0)  # (head_dim,)
                            grouped_queries.append(averaged_query)

                    q = torch.stack(grouped_queries, dim=0)  # (num_kv_heads, head_dim)

                else:
                    # Fewer query heads than KV heads (unusual but handle it)
                    # Repeat query heads to match KV heads
                    repeat_factor = num_kv_heads // num_q_heads
                    q_repeated = q.repeat_interleave(repeat_factor, dim=0)

                    # Handle any remaining heads
                    if q_repeated.shape[0] < num_kv_heads:
                        remaining = num_kv_heads - q_repeated.shape[0]
                        q_additional = q[:remaining]
                        q = torch.cat([q_repeated, q_additional], dim=0)
                    else:
                        q = q_repeated[:num_kv_heads]

            # Verify dimension matches what we expect from keys
            expected_dim = self.task_projector.head_dim
            if q.shape[1] != expected_dim:
                print(f"Warning: Query head dim {q.shape[1]} != expected key dim {expected_dim}")
                # Pad or truncate to match
                if q.shape[1] < expected_dim:
                    padding = torch.zeros(q.shape[0], expected_dim - q.shape[1],
                                          device=q.device, dtype=q.dtype)
                    q = torch.cat([q, padding], dim=1)
                else:
                    q = q[:, :expected_dim]

            return q  # (num_kv_heads, head_dim)
        """
        Extract query embedding from the last token that matches the key dimension.

        Args:
            input_ids: Input token IDs (B, T)
            layer_idx: Actual layer index in the model (not selected layers index)

        Returns:
            Query embedding from last token (head_dim,) - matching key dimension
        """
        with torch.no_grad():
            # Get hidden states up to the specified layer
            inputs = {"input_ids": input_ids, "output_hidden_states": True, "use_cache": False}
            outputs = self.model(**inputs)

            # Extract hidden state from the specified layer
            hidden_states = outputs.hidden_states[layer_idx + 1]  # +1 because outputs include embedding layer

            # Get the last token's hidden state
            last_token_hidden = hidden_states[0, -1, :]  # (d,) - take first batch, last token

            # Get attention layer
            if "gemma3" not in self.model.__class__.__name__.lower():
                attn_layer = self.model.model.layers[layer_idx].self_attn
            else:
                attn_layer = self.model.language_model.model.layers[layer_idx].self_attn

            # Apply input normalization first (same as in key extraction)
            if "qwen3" in self.model.__class__.__name__.lower():
                if "gemma3" not in self.model.__class__.__name__.lower():
                    normalized_hidden = self.model.model.layers[layer_idx].input_layernorm(
                        last_token_hidden.unsqueeze(0).unsqueeze(0))
                else:
                    normalized_hidden = self.model.language_model.model.layers[layer_idx].input_layernorm(
                        last_token_hidden.unsqueeze(0).unsqueeze(0))
                normalized_hidden = normalized_hidden[0, 0, :]  # (d,)
            elif "gemma" in self.model.__class__.__name__.lower():
                normalized_hidden = self.model.language_model.model.layers[layer_idx].input_layernorm(
                    last_token_hidden.unsqueeze(0).unsqueeze(0))
                normalized_hidden = normalized_hidden[0, 0, :]  # (d,)
            elif "llama" in self.model.__class__.__name__.lower():
                normalized_hidden = self.model.model.layers[layer_idx].input_layernorm(
                    last_token_hidden.unsqueeze(0).unsqueeze(0))
                normalized_hidden = normalized_hidden[0, 0, :]  # (d,)
            elif "mistral" in self.model.model.layers[layer_idx].__class__.__name__.lower():
                normalized_hidden = self.model.model.layers[layer_idx].input_layernorm(
                    last_token_hidden.unsqueeze(0).unsqueeze(0))
                normalized_hidden = normalized_hidden[0, 0, :]  # (d,)
            else:
                normalized_hidden = last_token_hidden

            # Extract query using the SAME process as key extraction
            # This ensures dimensional consistency
            if "qwen3" in self.model.__class__.__name__.lower():
                if hasattr(attn_layer, 'q_norm'):
                    input_shape = normalized_hidden.unsqueeze(0).unsqueeze(0).shape[:-1]  # (1, 1)
                    dim_h = self.model.config.head_dim
                    # Project and reshape: (1, 1, hidden) -> (1, 1, num_heads, head_dim)
                    q_proj = attn_layer.q_proj(normalized_hidden.unsqueeze(0).unsqueeze(0))
                    q_reshaped = q_proj.view(*input_shape, -1, dim_h)  # (1, 1, num_heads, head_dim)
                    # Apply q_norm if present
                    q = attn_layer.q_norm(q_reshaped)[0, 0]  # (num_heads, head_dim)
                else:
                    # Fallback: direct projection and reshape
                    q_proj = attn_layer.q_proj(normalized_hidden)  # (num_heads * head_dim,)
                    dim_h = self.model.config.head_dim
                    q = q_proj.view(-1, dim_h)  # (num_heads, head_dim)

            elif "gemma" in self.model.__class__.__name__.lower():
                if hasattr(attn_layer, 'q_norm'):
                    input_shape = normalized_hidden.unsqueeze(0).unsqueeze(0).shape[:-1]
                    dim_h = self.model.config.text_config.head_dim
                    q_proj = attn_layer.q_proj(normalized_hidden.unsqueeze(0).unsqueeze(0))
                    q_reshaped = q_proj.view(*input_shape, -1, dim_h)
                    q = attn_layer.q_norm(q_reshaped)[0, 0]  # (num_heads, head_dim)
                else:
                    q_proj = attn_layer.q_proj(normalized_hidden)
                    dim_h = self.model.config.text_config.head_dim
                    q = q_proj.view(-1, dim_h)

            elif "llama" in self.model.__class__.__name__.lower():
                input_shape = normalized_hidden.unsqueeze(0).unsqueeze(0).shape[:-1]
                hidden_shape = (*input_shape, -1, self.model.config.head_dim)
                q_proj = attn_layer.q_proj(normalized_hidden.unsqueeze(0).unsqueeze(0))
                q = q_proj.view(hidden_shape)[0, 0]  # (num_heads, head_dim)

            elif "mistral" in attn_layer.__class__.__name__.lower():
                input_shape = normalized_hidden.unsqueeze(0).unsqueeze(0).shape[:-1]
                hidden_shape = (*input_shape, -1, self.model.config.head_dim)
                q_proj = attn_layer.q_proj(normalized_hidden.unsqueeze(0).unsqueeze(0))
                q = q_proj.view(hidden_shape)[0, 0]  # (num_heads, head_dim)

            else:
                raise NotImplementedError(f"Unsupported model type: {self.model.__class__.__name__}")

            # For GQA models, we need to handle query/key head count mismatch
            # Use the first query head as representative (this matches key extraction logic)
            representative_query = q[0, :]  # (head_dim,)

            # Verify dimension matches what we expect from keys
            expected_dim = self.task_projector.head_dim
            if representative_query.shape[0] != expected_dim:
                print(f"Warning: Query dim {representative_query.shape[0]} != expected key dim {expected_dim}")
                # Pad or truncate to match
                if representative_query.shape[0] < expected_dim:
                    padding = torch.zeros(expected_dim - representative_query.shape[0],
                                          device=representative_query.device, dtype=representative_query.dtype)
                    representative_query = torch.cat([representative_query, padding], dim=0)
                else:
                    representative_query = representative_query[:expected_dim]

            return representative_query

    def attach_adaptive_projection(self,
                                   steer_mask_tensor=None,
                                   amplify_factor=None,
                                   silence=False):
        """Attach query-adaptive projection hooks"""
        self.remove_projection()

        amplify_factor = self.amplify_factor if amplify_factor is None else amplify_factor

        # Move steering mask to first device
        first_dev = self.device
        m_dev = (steer_mask_tensor if steer_mask_tensor is None
                 else steer_mask_tensor.unsqueeze(0) if steer_mask_tensor.dim() == 1
        else steer_mask_tensor).to(first_dev) if steer_mask_tensor is not None else None

        # Get model root
        root = self.model.module if hasattr(self.model, "module") else self.model

        # Pre-extract queries for each selected layer (more efficient than computing per hook)
        input_ids = getattr(self, '_current_input_ids', None)
        layer_queries = {}

        if input_ids is not None:
            for i, actual_layer in enumerate(self.sel_layers):
                try:
                    queries = self._extract_queries_from_context(input_ids, actual_layer)  # (num_heads, head_dim)
                    layer_queries[i] = queries.to(first_dev)
                except Exception as e:
                    print(f"Warning: Failed to extract queries for layer {actual_layer}: {e}")
                    # Fallback: will extract in hook
                    layer_queries[i] = None

        # Register hooks for selected layers
        for i, L in enumerate(self.sel_layers):
            attn = root.model.layers[L].self_attn if "gemma3" not in root.__class__.__name__.lower() else \
            root.language_model.model.layers[L].self_attn
            mod = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj

            # Move tensors to layer's device
            layer_device = next(mod.parameters()).device
            m_dev_layer = m_dev.to(layer_device) if m_dev is not None else None

            def _adaptive_hook(module, input_tensor, k_in,
                               m=m_dev_layer,
                               layer_idx=i,
                               actual_layer=L,
                               amp_factor=amplify_factor,
                               precomputed_query=layer_queries.get(i)):

                if k_in.dim() == 4:
                    B, T, H, D = k_in.shape
                    k_view = k_in
                elif k_in.dim() == 3:
                    B, T, D_all = k_in.shape
                    H = self.task_projector.num_heads
                    D = self.task_projector.head_dim
                    assert D_all == H * D, f"Dim mismatch: {D_all} != {H}*{D}"
                    k_view = k_in.view(B, T, H, D)
                else:
                    raise ValueError(f"Unsupported k_in shape: {k_in.shape}")

                if m is None or m.sum() == 0:
                    return k_in

                # Apply feature transformation
                k_feat = phi(k_view, self.feature_function)  # (B, T, H, D)

                # Use precomputed queries or extract from last token as fallback
                if precomputed_query is not None:
                    queries_all_heads = precomputed_query.to(k_feat.device)  # (num_heads, head_dim)
                else:
                    # Fallback: use last token key as approximation for all heads
                    print(f"Warning: Using fallback query extraction for layer {actual_layer}")
                    fallback_query = k_feat[0, -1, 0, :]  # (head_dim,) - use first head key as approximation
                    # Replicate for all heads
                    queries_all_heads = fallback_query.unsqueeze(0).repeat(H, 1)  # (num_heads, head_dim)

                # Apply adaptive projection per head
                for h in range(H):
                    if m.shape != (B, T):
                        continue  # Skip if mask shape doesn't match

                    # Get head-specific query
                    query_for_head = queries_all_heads[h, :]  # (head_dim,)

                    # Get dynamic projection for this head using head-specific query
                    dynamic_proj = self.task_projector.get_dynamic_projection(
                        query_for_head, layer_idx, h,
                        self.top_k_singular, self.combination_method
                    )  # (head_dim, head_dim)

                    # Apply dynamic projection to highlighted tokens
                    for b in range(B):
                        mask_b = m[b]  # (T,) - mask for batch b
                        if mask_b.sum() == 0:
                            continue  # No highlighted tokens in this batch

                        k_sel = k_feat[b, mask_b, h, :]  # (N_highlighted, head_dim)
                        if k_sel.numel() > 0:
                            # k' = k + amplify_factor * P_dynamic * k
                            # where P_dynamic = sum(α_m * P_m) with head-specific α_m coefficients
                            delta = torch.matmul(k_sel, dynamic_proj.T)  # (N_highlighted, head_dim)
                            k_feat[b, mask_b, h, :] += amp_factor * delta  # Apply amplification

                if k_in.dim() == 4:
                    return k_feat
                else:
                    return k_feat.contiguous().view(B, T, H * D)

            self._hooks.append(mod.register_forward_hook(_adaptive_hook))

        if not silence:
            print(f"✅ Adaptive steering hooks attached on layers {self.sel_layers}")
            print(f"   Using experts: {list(self.task_projector.expert_names)}")
            print(f"   Top-k singular vectors: {self.top_k_singular}")
            print(f"   Combination method: {self.combination_method}")
            print(f"   Amplify factor: {amplify_factor}")

    # Utility methods for configuration updates
    def remove_projection(self):
        """Remove all projection hooks"""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def eval(self):
        pass

    def train(self):
        pass

    def to(self, device):
        pass

    def set_amplify_factor(self, factor: float):
        """Update amplification factor"""
        self.amplify_factor = factor

    def set_combination_method(self, method: str):
        """Update combination method for singular vectors/values"""
        valid_methods = ["weighted_top_k", "all_weighted", "top_k_uniform"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        self.combination_method = method

    def set_top_k_singular(self, k: int):
        """Update number of top singular vectors to use"""
        self.top_k_singular = k

    def get_expert_info(self) -> dict:
        """Get information about loaded experts"""
        info = {}
        for expert_name in self.task_projector.expert_names:
            U_matrices = self.task_projector.expert_U_matrices[expert_name]
            singular_values = self.task_projector.expert_singular_values[expert_name]
            info[expert_name] = {
                "U_shape": tuple(U_matrices.shape),
                "S_shape": tuple(singular_values.shape),
                "num_parameters": U_matrices.numel() + singular_values.numel(),
                "device": str(U_matrices.device)
            }
        return info