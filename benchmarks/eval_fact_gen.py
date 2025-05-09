import os
import json
import logging
import argparse
import datasets
import transformers
from torch.utils.tensorboard import SummaryWriter

from benchmarks.counterfact.preprocess import load_dataset
from benchmarks.counterfact.evaluate import (
    counterfact_efficacy, 
    counterfact_paraphrase, 
    counterfact_generation
)

logger = logging.getLogger(__name__)

def main(args: argparse.Namespace):
    """Run the CounterFact benchmark."""
    datasets.disable_caching()
    
    # Initialize the model and tokenizer 
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    
    # Set up the evaluation data 
    logger.info("loading several data sources")
    dataset = load_dataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        attribute_no_entity=args.attribute_no_entity,
        example_subset=args.example_subset
    )
    
    results_output_dir = args.output_dir
    os.mkdir(results_output_dir, exist_ok=True, parents=True)
    tb_writter = SummaryWriter(log_dir=results_output_dir)
    
    logger.info(f"eval counterfact")
    for benchmark_name in args.benchmarks:
        results_file = os.path.join(results_output_dir, f"{benchmark_name}.json")
        if os.path.exists(results_file) and not args.overwrite_output_dir:
            logger.info(
                f"found existing {benchmark_name} results "
                f"at {results_file}"
            )
            continue
        
        if benchmark_name == "efficacy":
            results = counterfact_efficacy(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                batch_size=args.batch_size,
                n_top=args.n_top,
                max_length=args.max_length,
                add_unmediated_fact=args.add_unmediated_fact,
            )
        elif benchmark_name == "paraphrase":
            results = counterfact_paraphrase(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                batch_size=args.batch_size,
                max_length=args.max_length,
                add_unmediated_fact=args.add_unmediated_fact,
            )
        elif benchmark_name == "generation":
            results = counterfact_generation(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                batch_size=args.batch_size,
                max_length=args.max_length,
                add_unmediated_fact=args.add_unmediated_fact,
                attribute_snippets=None,
                tfidf_vectorizer=None
            )
        else:
            raise ValueError(f"unknown benchmark: {benchmark_name}")
        
        logging.info(
            f"{benchmark_name} benchmark complete! results:\n%s",
            json.dumps(results.metrics.to_dict(), indent=1),
        )
        for key, value in results.metrics.to_dict().items():
            tb_writter.add_scalar(f"{benchmark_name}/{key}", value['mean'], 1)
        
        with open(results_file, "w") as f:
            json.dump(results.to_dict(), f)

        metrics_file = os.path.join(results_output_dir, f"{benchmark_name}_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(results.metrics.to_dict(), f)
    
if __name__ == "__main__":
    BENCHMARKS = (
        "efficacy",
        "paraphrase",
        "generation",
    )
    
    parser = argparse.ArgumentParser(description="Evaluate CounterFact")
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
    
    parser.add_argument(
        "--benchmarks",
        "-b",
        nargs="+",
        choices=BENCHMARKS,
        default=BENCHMARKS,
        help="benchmarks to run, defaults depend on dataset",
    )
    
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--max_length", type=int, default=None, help="Max sequence length.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max generation length.")

    args = parser.parse_args()
    main(args)