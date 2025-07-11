from __future__ import annotations
import argparse, torch

from pastalib.profiler.synthetic_qa_builder import ProjectionBuilderBase
from src.utils import phi
import warnings
warnings.filterwarnings("ignore")

torch.set_grad_enabled(False)

class ProjectionBuilderKV(ProjectionBuilderBase):
    @staticmethod
    def extract_keys(model, tokenizer, text: str, indices: list[int], layers: list[int], feature: str) -> list[
        torch.Tensor]:
        result: list[torch.Tensor] = []
        
        def _hook(module, __, output: torch.Tensor):
            # (bsz, kv_heads, seq_len, hidden)
            if "gemma3" in str(module).lower():
                k_out = output
            elif "qwen3" in str(module).lower():
                k_out = output.transpose(1, 2)
            else:
                import pdb; pdb.set_trace()

            num_key_value_groups = model.config.num_attention_heads // model.config.num_key_value_heads
            k_out_repeated = torch.repeat_interleave(k_out, num_key_value_groups, dim=1).transpose(1, 2)[0]  # (seq_len, n_heads, h_dim)
            
            if torch.isnan(k_out_repeated).any():
                import pdb; pdb.set_trace()
                raise ValueError("h_out contains NaN values")
            
            k_out_repeated = phi(k_out_repeated.float(), feature).to(torch.float)
            
            # select only our tokens, and then return per-head slices
            k_out_sel = k_out_repeated[indices]  # (n_tokens, n_h, dim_h)
            for h in range(k_out_sel.size(1)):
                result.append(k_out_sel[:, h, :])

        hooks = [model.model.layers[L].self_attn.k_norm.register_forward_hook(_hook) for L in layers]

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

    builder = ProjectionBuilderKV(
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

