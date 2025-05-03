"""
python src/custom_builders/synthetic_qa_builder.py --model "pretrained/qwen2-1.5b-chat" --json "./data/synthetic/pair_qa.json"
"""

from src.model import ProjectionBuilderBase

import json
import random
import argparse

# ─────────────────── DATASET‑SPECIFIC BUILDER ─────────────────────────
class SynthQABuilder(ProjectionBuilderBase):
    """
    Synthetic QA dataset where each record is a dict with
       • relevant_context
       • answer
       • irrelevant_context
       • question
    """

    def __init__(self, *, json_file: str, **kwargs):
        super().__init__(**kwargs)
        with open(json_file) as f:
            self.data = json.load(f)
        random.shuffle(self.data)

    def iter_examples(self):
        for ex in self.data:
            rel_ctx   = ex["relevant_context"]
            rel_tok   = ex["answer"]
            irrel_ctx = ex["irrelevant_context"]
            q         = ex["question"]

            ctx = (f"{rel_ctx}\n{irrel_ctx}\nQuestion: {q}"
                   if random.random() < .5
                   else f"{irrel_ctx}\n{rel_ctx}\nQuestion: {q}")

            yield ctx, rel_tok, irrel_ctx


# ───────────────────────── CLI entry ‑ point ──────────────────────────
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument('--model',   required=True)
    pa.add_argument('--layers',  default='all')
    pa.add_argument('--json',    required=True)
    pa.add_argument('--samples', type=int, default=100)
    pa.add_argument('--top-pct', type=float, default=0.97)
    args = pa.parse_args()

    builder = SynthQABuilder(
        model_path=args.model,
        layers=args.layers,
        top_pct=args.top_pct,
        max_samples=args.samples,
        json_file=args.json
    )
    builder.run(
        pos_out=f"projections/synthetic/{args.model.split('/')[-1]}_pos_proj.pt",
        neg_out=f"projections/synthetic/{args.model.split('/')[-1]}_neg_proj.pt",
    )  # writes projections exactly as before
