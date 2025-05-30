import os
import json
import logging
import argparse
import datasets
import transformers
from pathlib import Path

from src.model import SEKALLM
from pastalib.pasta import PASTA, read_head_config

from benchmarks.counterfact.preprocess import load_dataset
from benchmarks.counterfact.evaluate import (
    counterfact_efficacy, 
    counterfact_paraphrase, 
    counterfact_generation,
    load_attribute_snippets,
    load_counterfact_tfidf_vectorizer
)
from benchmarks.utils.pasta_utils import setup_logger

logger = logging.getLogger(__name__)

def main(args: argparse.Namespace):
    """Run the CounterFact benchmark."""
    datasets.disable_caching()
    logger.info(args)
    
    # Initialize the model and tokenizer
    pasta = None
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
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model,
            padding_side="left"
        )
        if args.pasta:
            head_config = read_head_config(args.head_config)
            pasta = PASTA(
                model, 
                tokenizer,
                head_config=head_config, 
                alpha=args.pasta_alpha, 
                scale_position=args.scale_position,
            )
    
    # Set up the evaluation data
    data_path = os.path.join(args.data_path, "counterfact.jsonl")    
    logger.info("loading several data sources")
    dataset = load_dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        attribute_no_entity=args.attribute_no_entity,
        example_subset=args.example_subset
    )
    
    results_output_dir = Path(args.output_dir)
    results_output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"eval counterfact")
    if args.marker_end is None:
        args.marker_end = args.marker_start
    for benchmark_name in args.benchmarks:
        results_file = results_output_dir / f"{benchmark_name}.json"
        if results_file.exists() and not args.overwrite_output_dir:
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
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                add_unmediated_fact=args.add_unmediated_fact,
                chat=args.chat,
                seka=args.seka,
                pasta=pasta,
                add_marker=args.add_marker,
                marker_start=args.marker_start,
                marker_end=args.marker_end,
            )
        elif benchmark_name == "paraphrase":
            results = counterfact_paraphrase(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                batch_size=args.batch_size,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                add_unmediated_fact=args.add_unmediated_fact,
                chat=args.chat,
                seka=args.seka,
                pasta=pasta,
                add_marker=args.add_marker,
                marker_start=args.marker_start,
                marker_end=args.marker_end,
            )
        elif benchmark_name == "generation":
            snippets_file = os.path.join(args.data_path, "attribute_snippets.json")
            idf_file = os.path.join(args.data_path, "idf.npy")
            vocab_file = os.path.join(args.data_path, "tfidf_vocab.json")
            results = counterfact_generation(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                batch_size=args.batch_size,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                add_unmediated_fact=args.add_unmediated_fact,
                attribute_snippets=load_attribute_snippets(snippets_file),
                tfidf_vectorizer=load_counterfact_tfidf_vectorizer(idf_file, vocab_file),
                chat=args.chat,
                seka=args.seka,
                pasta=pasta,
                add_marker=args.add_marker,
                marker_start=args.marker_start,
                marker_end=args.marker_end,
            )
        else:
            raise ValueError(f"unknown benchmark: {benchmark_name}")
        
        logging.info(
            f"{benchmark_name} benchmark complete! results:\n%s",
            json.dumps(results.metrics.to_dict(), indent=1),
        )
        
        with results_file.open("w") as f:
            json.dump(results.to_dict(), f, indent=4)

        metrics_file = results_output_dir / f"{benchmark_name}_metrics.json"
        with metrics_file.open("w") as f:
            json.dump(results.metrics.to_dict(), f, indent=4)
    
if __name__ == "__main__":
    setup_logger()
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
        help="Path to the data"
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
    parser.add_argument("--add_unmediated_fact", type=bool, default=True, help="Present models both facts.")
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
    
    parser.add_argument("--pasta", action="store_true", default=False, help="Use PASTA model")
    parser.add_argument("--head_config", type=str, default=None, help="PASTA head config for steering")
    parser.add_argument("--pasta_alpha", type=float, default=None, help="Scaling coefficient")
    parser.add_argument("--scale_position", type=str, default=None, help="Steer the selected section or others")

    args = parser.parse_args()
    main(args)