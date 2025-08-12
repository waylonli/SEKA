# src/custom_builders/synthetic_qa_builder.py
from src.model.projection_builder_base_adapsvd import ProjectionBuilderBase

import random
import argparse

import torch

from benchmarks.biasbios.preprocess import load_dataset
from benchmarks.utils.pasta_utils import get_column_names

from typing import cast

# set random seed
random.seed(42)


class BiasBiosQABuilder(ProjectionBuilderBase):
    def __init__(
        self, 
        model_path, 
        data_path, 
        example_subset, 
        layers, 
        top_pct, 
        feature, 
        max_samples, 
        min_diff, 
        chat = False, 
        device = 'cuda' if torch.cuda.is_available() else 'cpu', 
        save_svd = False, 
        save_traditional = True
    ):
        super().__init__(model_path, data_path, layers, top_pct, feature, max_samples, min_diff, chat, device, save_svd, save_traditional)
        self.example_subset = example_subset
        
    # def assemble_texts(self, ctx: str, rel_q: str):
    #     if self.chat:
    #         text_H = self.tokenizer.apply_chat_template([{"role": "user", "content": f"Context: {ctx}"}],
    #                                                     tokenize=False)
    #         text_Hp = self.tokenizer.apply_chat_template(
    #             [{"role": "user", "content": f"Question: {rel_q}\nContext: {ctx}"}], tokenize=False)
    #     else:
    #         text_H, text_Hp = f"Context: {ctx} ", f"Question: {rel_q}\nContext: {ctx}"

    #     return text_H, text_Hp
    
    def iter_examples(self):
        # load dataset
        dataset = load_dataset(self.data_path, False, self.example_subset)
        columns = get_column_names(dataset, exclude=["target_unmediated"])
        with dataset.formatted_as(columns=columns):
            loader = torch.utils.data.DataLoader(dataset, batch_size=1)
            for batch in loader:
                if "gemma" in self.model.__class__.__name__.lower():
                    batch["prompt"] = [p.strip() for p in batch["prompt"]]
                elif hasattr(self.model, "model") and "gemma3" in self.model.model.__class__.__name__.lower():
                    batch["prompt"] = [p.strip() for p in batch["prompt"] if p.strip()]
                
                prompt = batch["prompt"][0]
                context = batch["context"][0]
                target_mediated = batch["target_mediated"][0]
                
                question = prompt[len(context):].strip() + " ?"

                yield {"context_1": context, "question_1": question, "answer_1": target_mediated}
                
                # # short context
                # main_context = context.split(".")[0].strip() + "."
                # yield {"context_1": main_context, "question_1": question, "answer_1": target_mediated}

    def get_triplets(self, ex: dict) -> list[tuple[str, str, str]]:
        return [
            (ex['context_1'], ex['question_1'], ex['answer_1'])
        ]


if __name__ == '__main__':
    """
    Usage:
    python src/custom_builders/biasbios_qa_builder.py \
    --model <model-path> \
    --data <data-path-to-json> \
    --example_subset 0:200 \
    --max_samples 200 \
    --output_dir <output-path> \
    --min_diff 1e-6
    """
    
    parser = argparse.ArgumentParser(description="Build projections from synthetic QA data")
    parser.add_argument('--model', required=True, help='Model path or HF identifier')
    parser.add_argument('--data', required=True, help='Path to QA file')
    parser.add_argument("--example_subset", type=str, default="0:200", help="run on a subset of data")
    parser.add_argument('--layers', default='all', help='Layers to use for projection')
    parser.add_argument('--top_pct', type=float, default=0.9,
                        help='Percentage of variance to retain in SVD')
    parser.add_argument('--feature', type=str, default=None,
                        help='Feature function to apply (tanh, elu, squared-exponential)')
    parser.add_argument('--max_samples', type=int, default=200,
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
    print("Building BiasBios QA projections with configuration:")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}[{args.example_subset}]")
    print(f"  Layers: {args.layers}")
    print(f"  Feature function: {args.feature}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Output formats:")
    print(f"    - Traditional projections: {args.save_traditional}")
    print(f"    - SVD components: {args.save_svd}")

    builder = BiasBiosQABuilder(
        model_path=args.model,
        data_path=args.data,
        example_subset=args.example_subset,
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
