import os
import argparse
import json
import logging
import datasets
import transformers
from pathlib import Path

from benchmarks.biosbias.preprocess import load_dataset
from benchmarks.biosbias.evaluate import biasbios_prediction_evaluation
from benchmarks.utils.pasta_utils import setup_logger

from src.model import SEKALLM

logger = logging.getLogger(__name__)

def main(args: argparse.Namespace):
    """Run the evaluation for instruction following tasks."""
    datasets.disable_caching()
    
    # Initialize the model and tokenizer 
    if args.seka:
        if "_tanh" in args.pos:
            feature_fn = "tanh"
        elif "_elu" in args.pos:
            feature_fn = "elu"
        elif "_squared" in args.pos:
            feature_fn = "squared-exponential"
        else:
            feature_fn = None

        model = SEKALLM(
            args.model,
            pos_pt=args.pos,
            neg_pt=args.neg,
            marker_start=args.marker_start,
            marker_end=args.marker_end,
            layers=args.layers,
            amplify_pos=args.amplify_pos,
            amplify_neg=args.amplify_neg,
            feature_function=feature_fn,
            torch_dtype="auto",
            device="auto"
        )
        tokenizer = model.tok
            
        # Force add_marker flag to be True
        if not args.add_marker:
            logger.warning("SEKA LLM requires markers, setting add_marker to True.")
            args.add_marker = True
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, 
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, padding_side="left")
    
    # Set up the evaluation data 
    dataset = load_dataset(args.data_path, args.attribute_no_entity, args.example_subset)

    result_output_dir = Path(args.output_dir)
    result_output_dir.mkdir(exist_ok=True, parents=True) 
    result_output_file = result_output_dir / "result.json"
    
    if not os.path.exists(result_output_file) or args.overwrite_output_dir:
        logger.info("begin evaluation")
        
        if args.add_marker and args.marker_end is None:
            args.marker_end = args.marker_start
        
        results = biasbios_prediction_evaluation(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            data_path=args.data_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            desc="BiasBios Evaluation",
            add_marker=args.add_marker,
            marker_start=args.marker_start,
            marker_end=args.marker_end,
            seka=args.seka,
        )
        logging.info(
            f"Evaluation complete! results:\n%s",
            json.dumps(results.metrics.to_dict(), indent=1),
        )
        # Readout the results
        with result_output_file.open("w") as f:
            json.dump(results.to_dict(), f, indent=4)
    else:
        logger.info(
            f"existing results found at {result_output_file}; skipping"
        )

if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser(
        description="Evaluation model generation on BiasBios dataset."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="model name or path",
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True, 
        help="Path to the dataset"
    )
    parser.add_argument(
        "--attribute-no-entity",
        action="store_true",
        default=False,
        help="set context = attribute",
    )
    parser.add_argument(
        "--example_subset", type=str, default=None, help="run on a subset of data"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="unique name for the experiment",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true", help="")
    
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--max_length", type=int, default=None, help="Max sequence length.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max generation length.")

    parser.add_argument("--chat", action="store_true", default=False, help="Apply chat template")
    parser.add_argument("--add_marker", action="store_true", default=False, help="Apply marked prompting")
    parser.add_argument('--marker_start', default='**',
                help='highlight start marker (e.g. 👉 )')
    parser.add_argument('--marker_end', default=None,
                help='highlight end marker; defaults to same as start')

    parser.add_argument("--seka", action="store_true", default=False, help="Use SEKA model")
    parser.add_argument('--pos', type=str, default=None,
                    help='positive (relevant) projector .pt')
    parser.add_argument('--neg', type=str, default=None,
                help='optional negative (irrelevant) projector .pt')
    parser.add_argument('--amplify_pos', default=1.5, type=float)
    parser.add_argument('--amplify_neg', default=0.5, type=float)
    parser.add_argument('--layers', default='last10',
                help="'all' / 'last4' / '0,4,19' …")
    
    args = parser.parse_args()
    main(args)