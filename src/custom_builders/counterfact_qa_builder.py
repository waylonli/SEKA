# src/custom_builders/synthetic_qa_builder.py
from src.model.projection_builder_base_pos_check import ProjectionBuilderBase

import os
from functools import partial
import random
import argparse

import torch

from benchmarks.counterfact.preprocess import load_dataset, precompute_token_ids
from benchmarks.counterfact.evaluate import _counterfact_select_and_flatten

# set random seed
random.seed(42)


class CounterfactQABuilder(ProjectionBuilderBase):
    def __init__(
        self, 
        model_path, 
        data_path, 
        example_subset, 
        benchmark_name,
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
        self.benchmark_name = benchmark_name
        
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
        data_path = os.path.join(self.data_path, "counterfact.jsonl")    
        dataset = load_dataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            attribute_no_entity=False,
            example_subset=self.example_subset
        )
        
        if self.benchmark_name == "paraphrase":
            dataset = _counterfact_select_and_flatten(
                dataset, "paraphrase_prompts"
            )
            dataset = dataset.map(
                partial(precompute_token_ids,
                    tokenizer=self.tokenizer,
                    target_token_first_space=False,
                ),
                batched=True,
                batch_size=64,
                keep_in_memory=True,
                num_proc=1,
            )
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        for batch in loader:
            prompt = batch["prompt"][0]
            context = batch["context"][0]
            attribute = batch["attribute"][0]
            target_mediated = batch["target_mediated"][0]
            target_unmediated = batch["target_unmediated"][0]

            unmediated_fact = context.replace(target_mediated, target_unmediated)
            question = prompt[len(context)+1:].strip() + " ?"
            full_context = f"Previously, {unmediated_fact}. Currently, {context}."
            
            yield {"context_1": full_context, "question_1": question, "answer_1": target_mediated}

        
    def get_triplets(self, ex: dict) -> list[tuple[str, str, str]]:        
        return [
            (ex['context_1'], ex['question_1'], ex['answer_1'])
        ]


if __name__ == '__main__':
    """
    Usage:
    python src/custom_builders/counterfact_qa_builder.py \
    --model <model-path> \
    --data <data-path-to-json> \
    --benchmark_name efficacy \
    --example_subset 0:200 \
    --max_samples 200 \
    --output_dir <output-path> \
    --min_diff 1e-6
    """
    
    BENCHMARKS = (
        "efficacy",
        "paraphrase",
    )
    
    parser = argparse.ArgumentParser(description="Build projections from synthetic QA data")
    parser.add_argument('--model', required=True, help='Model path or HF identifier')
    parser.add_argument('--data', required=True, help='Path to QA file')
    parser.add_argument("--example_subset", type=str, default="0:200", help="run on a subset of data")
    parser.add_argument("--benchmark_name", choices=BENCHMARKS, default=BENCHMARKS[0], help="benchmarks to run, defaults depend on dataset",)
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
    print("Building Counterfact QA projections with configuration:")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}[{args.example_subset}]")
    print(f"  Benchmark: {args.benchmark_name}")
    print(f"  Layers: {args.layers}")
    print(f"  Feature function: {args.feature}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Output formats:")
    print(f"    - Traditional projections: {args.save_traditional}")
    print(f"    - SVD components: {args.save_svd}")

    builder = CounterfactQABuilder(
        model_path=args.model,
        data_path=args.data,
        example_subset=args.example_subset,
        benchmark_name=args.benchmark_name,
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
