import os
import argparse
import json
import logging
import datasets
import transformers
from torch.utils.tensorboard import SummaryWriter

from benchmarks.biasbios.preprocess import load_dataset
from benchmarks.biasbios.evaluate import biasbios_prediction_evaluation
from benchmarks.utils.pasta_utils import setup_logger

logger = logging.getLogger(__name__)

def main(args: argparse.Namespace):
    """Run the evaluation for instruction following tasks."""
    datasets.disable_caching()
    
    # Initialize the model and tokenizer 
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    
    # Set up the evaluation data 
    dataset = load_dataset(args.data_path, args.attribute_no_entity, args.example_subset)

    # TODO: Set up the output dir 
    result_output_dir = args.output_dir
    os.mkdir(result_output_dir, exist_ok=True, parents=True) 
    result_output_file = result_output_dir / "result.json"
    
    if not os.path.exists(result_output_file) or args.overwrite_output_dir:
        logger.info("begin evaluation")
        results = biasbios_prediction_evaluation(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            data_path=args.data_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            desc="BiasBios Evaluation",
        )
        logging.info(
            f"Evaluation complete! results:\n%s",
            json.dumps(results.metrics.to_dict(), indent=1),
        )
        # Readout the results
        tb_writter = SummaryWriter(log_dir=str(result_output_dir))
        metrics = results.metrics.to_dict() 
        tb_writter.add_scalar("top1_accuracy", metrics['top1_accuracy'], 1)
        tb_writter.add_scalar(f"top{metrics['k']}_accuracy", metrics['topk_accuracy'], 1)
        for key in ["mean", "std"]:
            tb_writter.add_scalar(f"fluency/{key}", metrics['fluency'][key], 1)
            tb_writter.add_scalar(f"consistency/{key}", metrics['consistency'][key], 1)

        with open(result_output_file, "w") as f:
            json.dump(results.to_dict(), f)
        tb_writter.close()
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
    
    args = parser.parse_args()
    main(args)