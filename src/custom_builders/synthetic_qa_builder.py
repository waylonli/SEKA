"""
python src/custom_builders/synthetic_qa_builder.py --model "pretrained/qwen2-1.5b-chat" --json "./data/synthetic/pair_qa.json"
"""

from src.model import ProjectionBuilderBase

import json
import random
import argparse
# set random seed
random.seed(42)

# ─────────────────── DATASET‑SPECIFIC BUILDER ─────────────────────────
class SynthQABuilder(ProjectionBuilderBase):
    """
    Synthetic QA dataset where each record is a dict with
       • relevant_context
       • answer
       • irrelevant_context
       • question
    """

    def __init__(self, *, json_file: str, chat: bool, **kwargs):
        super().__init__(**kwargs)
        with open(json_file) as f:
            self.data = json.load(f)
        self.chat = chat
        random.shuffle(self.data)

    def iter_examples(self):
        for ex in self.data:
            rel_ctx   = ex["relevant_context"]
            rel_tok   = ex["answer"]
            irrel_ctx = ex["irrelevant_context"]
            q         = ex["question"] if self.chat else ex["base_postfix"]

            if not self.chat:
                ctx = (f"{rel_ctx}\n{irrel_ctx}\n{q + rel_tok}"
                       if random.random() < .5
                       else f"{irrel_ctx}\n{rel_ctx}\n{q + rel_tok}")
            else:
                tokens = self.tok.apply_chat_template(
                    [{
                        "role": "user",
                        "content": f"{rel_ctx}\n{irrel_ctx}\nQuestion: {q}"
                    },
                    {
                        "role": "assistant",
                        "content": "Answer: " + rel_tok
                    }
                    ],
                )
                ctx = self.tok.decode(tokens, skip_special_tokens=False)

            yield ctx, rel_tok, irrel_ctx


# ───────────────────────── CLI entry ‑ point ──────────────────────────
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument('--model',   required=True)
    pa.add_argument('--layers',  default='all')
    pa.add_argument('--json',    required=True)
    pa.add_argument('--samples', type=int, default=100)
    pa.add_argument('--top-pct', type=float, default=0.97)
    pa.add_argument('--chat',    action='store_true',
                      help="use chat template for context")
    pa.add_argument('--feature-function', type=str, default=None)

    args = pa.parse_args()

    builder = SynthQABuilder(
        model_path=args.model,
        layers=args.layers,
        top_pct=args.top_pct,
        max_samples=args.samples,
        json_file=args.json,
        chat=args.chat,
        feature_function=args.feature_function,
    )
    builder.run(
        pos_out=f"projections/synthetic/{args.model.split('/')[-1]}_pos_proj.pt" if args.feature_function is None else f"projections/synthetic/{args.model.split('/')[-1]}_pos_proj_{args.feature_function}.pt",
        neg_out=f"projections/synthetic/{args.model.split('/')[-1]}_neg_proj.pt" if args.feature_function is None else f"projections/synthetic/{args.model.split('/')[-1]}_neg_proj_{args.feature_function}.pt",
    )  # writes projections exactly as before
