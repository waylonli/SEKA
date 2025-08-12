# src/custom_builders/synthetic_qa_builder.py
from src.model.projection_builder_base_adapsvd import ProjectionBuilderBase

import json
import random
import argparse

# set random seed
random.seed(42)


class SynthQABuilder(ProjectionBuilderBase):
    def iter_examples(self):
        with open(self.data_path) as f:
            for i, line in enumerate(f):
                if i >= self.max_samples:
                    break
                yield json.loads(line)

    def get_triplets(self, ex: dict) -> list[tuple[str, str, str, str]]:
        return [
            (ex['context_1'], ex['question_1'], ex['answer_1'])
        ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build projections from synthetic QA data")
    parser.add_argument('--model', required=True, help='Model path or HF identifier')
    parser.add_argument('--data', required=True, help='Path to synthetic QA JSON file')
    parser.add_argument('--layers', default='all', help='Layers to use for projection')
    parser.add_argument('--top_pct', type=float, default=0.9,
                        help='Percentage of variance to retain in SVD')
    parser.add_argument('--feature', type=str, default=None,
                        help='Feature function to apply (tanh, elu, squared-exponential)')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum number of samples to process')
    parser.add_argument('--min_diff', type=float, default=2.0,
                        help='Minimum norm difference threshold for applying projection')
    parser.add_argument('--chat', action='store_true',
                        help='Apply chat template to prompts')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for projections')

    # NEW: Output format options
    parser.add_argument('--save-svd', action='store_true',
                        help='Save SVD components (U matrices and singular values)')
    parser.add_argument('--save-traditional', action='store_true', default=True,
                        help='Save traditional projection matrices (default: True)')
    parser.add_argument('--svd-only', action='store_true',
                        help='Save only SVD components (equivalent to --save-svd --no-save-traditional)')

    args = parser.parse_args()

    # Handle svd-only flag
    if args.svd_only:
        args.save_svd = True
        args.save_traditional = False

    # Ensure at least one output format is selected
    if not args.save_svd and not args.save_traditional:
        print("Warning: No output format selected. Defaulting to traditional projections.")
        args.save_traditional = True

    # Log the configuration
    print("Building synthetic QA projections with configuration:")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Layers: {args.layers}")
    print(f"  Feature function: {args.feature}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Output formats:")
    print(f"    - Traditional projections: {args.save_traditional}")
    print(f"    - SVD components: {args.save_svd}")

    builder = SynthQABuilder(
        model_path=args.model,
        data_path=args.data,
        layers=args.layers,
        top_pct=args.top_pct,
        feature=args.feature,
        max_samples=args.max_samples,
        min_diff=args.min_diff,
        chat=args.chat,
        save_svd=args.save_svd,  # NEW
        save_traditional=args.save_traditional  # NEW
    )

    builder.run(args.output_dir)

    print(f"\n🎉 Synthetic QA projection building complete!")
    print(f"   Output directory: {args.output_dir}")

    if args.save_traditional:
        print(f"   Traditional files: *_pos_proj.pt, *_neg_proj.pt")
    if args.save_svd:
        print(f"   SVD files: *_pos_svd.pt, *_neg_svd.pt")

    # norm_diff_14b = torch.load(os.path.join(args.output_dir, 'norm_diffs_Qwen3-14B-Base.pt'), weights_only=False)
    # SynthQABuilder.plot_norm_heatmap(norm_diff_14b, "Qwen3-14B-Base", range(40), args.output_dir)
