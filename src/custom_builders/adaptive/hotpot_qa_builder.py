# src/custom_builders/synthetic_qa_builder.py
from src.model.projection_builder_base_adapsvd import ProjectionBuilderBase

import json
import random
import argparse

# set random seed
random.seed(42)


class HotpotQABuilder(ProjectionBuilderBase):
    @staticmethod
    def span_token_indices(tokenizer, text: str, subs: list[str]) -> list[int] | None:
        span_indices = []
        for sub in subs:
            low, sub_low = text.lower(), sub.lower()
            if sub_low not in low:
                return None
            start = low.index(sub_low)
            end = start + len(sub_low)
            enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            indices = [i for i, (s, e) in enumerate(enc.offset_mapping) if s >= start and e <= end]
            if len(indices) == 0:
                indices = [i for i, (s, e) in enumerate(enc.offset_mapping) if s >= (start - 1) and e <= end]
            if len(indices) == 0:
                indices = [i for i, (s, e) in enumerate(enc.offset_mapping) if s >= start and e <= (end + 1)]
            span_indices.extend(indices)

        return span_indices
    
    def iter_examples(self):
        with open(args.data) as f:
            data = json.load(f)
        self.max_samples = min(self.max_samples, len(data))
        data = data[:self.max_samples]
        
        for item in data:
            question = item["question"]
            answer = item["answer"]
            
            # combine context list
            context_map = {}
            full_context = ""
            delimiter = ""
            for title, snts in item["context"]:
                cur_cxt = f"{title} {''.join(snts)}"
                full_context += delimiter + cur_cxt
                delimiter = " "
                context_map[title] = snts
            
            supporting_facts = []
            for title, snt_idx in item["supporting_facts"]:
                supporting_facts.append(context_map[title][snt_idx])

            yield {"context_1": full_context, "question_1": question, "answer_1": supporting_facts}
        
    def get_triplets(self, ex: dict) -> list[tuple[str, str, str]]:
        return [
            (ex['context_1'], ex['question_1'], ex['answer_1'])
        ]

        
if __name__ == '__main__':
    """
    Usage:
    python src/custom_builders/hotpot_qa_builder.py \
    --model <model-path> \
    --data <data-path-to-json> \
    --max_samples 200 \
    --output_dir <output-path> \
    --min_diff 1e-6
    """
    
    parser = argparse.ArgumentParser(description="Build projections from synthetic QA data")
    parser.add_argument('--model', required=True, help='Model path or HF identifier')
    parser.add_argument('--data', required=True, help='Path to QA file')
    parser.add_argument('--layers', default='all', help='Layers to use for projection')
    parser.add_argument('--top_pct', type=float, default=0.9,
                        help='Percentage of variance to retain in SVD')
    parser.add_argument('--feature', type=str, default=None,
                        help='Feature function to apply (tanh, elu, squared-exponential)')
    parser.add_argument('--max_samples', type=int, default=90447,
                        help='Maximum number of samples to process')
    parser.add_argument('--min_diff', type=float, default=0.2,
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
    print("Building Hotpot QA projections with configuration:")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Layers: {args.layers}")
    print(f"  Feature function: {args.feature}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Output formats:")
    print(f"    - Traditional projections: {args.save_traditional}")
    print(f"    - SVD components: {args.save_svd}")

    builder = HotpotQABuilder(
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
        print(f"   Traditional files: *_pos_proj.pt")
    if args.save_svd:
        print(f"   SVD files: *_pos_svd.pt")