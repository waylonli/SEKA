import os
import argparse
import json
import logging
import datasets
import transformers
# from torch.utils.tensorboard import SummaryWriter

from benchmarks.biasbios.preprocess import load_dataset
from benchmarks.biasbios.evaluate import biasbios_instruction_evaluation, BiosBiasInstructionEvaluationResults

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
    logger.info("loading several data sources")
    dataset = load_dataset(
        data_path=args.data_path,
        attribute_no_entity=args.attribute_no_entity,
        example_subset=args.example_subset
    )
    
    evaluation_result_dir = args.output_dir
    os.mkdir(evaluation_result_dir, exist_ok=True, parents=True) 
    result_file = evaluation_result_dir / f"instruction_evaluation_{args.task}.json"
    metric_file = evaluation_result_dir / "metric_result.json" 

    if not os.path.exists(result_file) or args.overwrite_output_dir:
        logger.info("begin baseline")
        evluation_result: BiosBiasInstructionEvaluationResults = biasbios_instruction_evaluation(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            task=args.task, 
            prompt_idx=args.prompt_idx, 
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            desc="Instruction evaluation [LM]",
        )
        logging.info(
            f"Evaluation complete! results:\n%s",
            json.dumps(evluation_result.metrics.to_dict(), indent=1),
        )
        # tb_writter = SummaryWriter(log_dir=evaluation_result_dir)

        metrics = evluation_result.metrics.to_dict() 
        # tb_writter.add_scalar("top1_accuracy", metrics['top1_accuracy'], 1)
        # tb_writter.add_scalar(f"top{metrics['k']}_accuracy", metrics['topk_accuracy'], 1)
        # for key in ["mean", "std"]:
        #     tb_writter.add_scalar(f"fluency/{key}", metrics['fluency'][key], 1)
        #     tb_writter.add_scalar(f"consistency/{key}", metrics['consistency'][key], 1)
        instruction_evaluation_result = metrics['instruction_evaluation'] 
        # for key in instruction_evaluation_result:
        #     tb_writter.add_scalar(f"instruction_evaluation/{key}", instruction_evaluation_result[key], 1)
        # tb_writter.close()
        
        with open(result_file, "w") as f:
            json.dump(evluation_result.to_dict(), f)
        with open(metric_file, "w") as f:
            json.dump(metrics, f) 
    else:
        logger.info(
            f"existing baseline results found at {result_file}; skipping"
        )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate generation on bias dataset"
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
    parser.add_argument(
        "--task", type=str, default="json", help="The name of evaluation task: [json|pronchange]."
    )
    parser.add_argument(
        "--prompt_idx", nargs="+", type=int, default=0, help="Which prompt template to apply for evaluation."
    )
    
    args = parser.parse_args()
    main(args)