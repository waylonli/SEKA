from src.model import ProjectionBuilderBase

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

    def get_triplets(self, ex: dict) -> list[tuple[str,str,str,str]]:
        return [
            (ex['context_1'], ex['question_1'], ex['answer_1'], ex['question_2']),
            (ex['context_2'], ex['question_2'], ex['answer_2'], ex['question_1'])
        ]

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

    builder = SynthQABuilder(
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

    # norm_diff_14b = torch.load(os.path.join(args.output_dir, 'norm_diffs_Qwen3-14B-Base.pt'), weights_only=False)
    # SynthQABuilder.plot_norm_heatmap(norm_diff_14b, "Qwen3-14B-Base", range(40), args.output_dir)
