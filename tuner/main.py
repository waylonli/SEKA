import argparse
import optuna
import json
import datasets


MARKER_START = "**" 
MARKER_END = "**" 
LAYERS = "all"


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Builder args
    build_methods = ["synth", "synth_ada", "efficacy", "paraphrase", "biabios", "hotpotqa"]
    parser.add_argument("--build_method", type=str, choices=build_methods, 
                        help="Build methods")
    parser.add_argument("--build_data_path", type=str, 
                        help="Path to QA file")
    parser.add_argument("--build_example_subset", type=str, default=None,
                        help="build on a subset of data")
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of samples to process')
    parser.add_argument('--save_svd', action='store_true',
                        help='Save SVD components (U matrices and singular values)')
    parser.add_argument('--save_traditional', action='store_true', default=True,
                        help='Save traditional projection matrices (default: True)')
    parser.add_argument('--proj_dir', type=str, default="tuner/tmp",
                        help='Output directory for projections')
    
    # Model args
    model_types = ["seka", "ada_seka", "varnorm_seka"]
    parser.add_argument("--model_type", type=str, choices=model_types, 
                        help="Model type")
    parser.add_argument("--model_path", type=str,
                        help="Model path or HF identifier")
    parser.add_argument('--chat', action='store_true',
                        help='Apply chat template to prompts')
    
    # Eval args
    tasks = ["counterfact", "biasbios", "pronchange"]
    parser.add_argument("--task", type=str, choices=tasks,
                        help="Task to tune")
    parser.add_argument("--eval_data_path", type=str,
                        help="Path to evaluation dataset")
    parser.add_argument("--eval_example_subset", type=str, default="4500:5000",
                        help="run on a subset of data")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size.")
    parser.add_argument("--max_length", type=int, default=None, 
                        help="Max sequence length.")
    parser.add_argument("--max_new_tokens", type=int, default=None, 
                        help="Max generation length.")
    
    # SEKA args
    parser.add_argument("--use_proj_neg", action='store_true',
                        help="Use negative projection")
    
    # AdaSEKA args
    parser.add_argument("--top_k_singular", type=int, default=5,
                        help="Top k singular values for adaptive SEKA")
    parser.add_argument("--combination_method", type=str, default="weighted_top_k", choices=["weighted_top_k", "all_weighted", "top_k_uniform"], 
                        help="Combination method for adaptive SEKA")
    
    # Params to tune
    parser.add_argument('--n_trials', type=int, default=50,
                        help="Number of trials")
    parser.add_argument('--top_pct_range', nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help='[Builder] The [min, max] range for top_pct hyperparameter')
    parser.add_argument('--min_diff_range', nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help='[Builder] The [min, max] range for the min_diff hyperparameter.')
    parser.add_argument('--amp_pos_range', nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help='[SEKA] The [min, max] range for the amplify_pos hyperparameter.')
    parser.add_argument('--amp_neg_range', nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help='[SEKA] The [min, max] range for the amplify_neg hyperparameter.')
    parser.add_argument('--amp_factor_range', nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help='[AdaSEKA] The [min, max] range for the amplify_factor hyperparameter.')
    
    parser.add_argument('--fixed_top_pct', type=float, default=None, 
                        help='Set a fixed value for top_pct and skip tuning it.')
    parser.add_argument('--fixed_min_diff', type=float, default=None, 
                        help='Set a fixed value for min_diff and skip tuning it.')
    parser.add_argument('--fixed_amp_pos', type=float, default=None, 
                        help='Set a fixed value for amp_pos and skip tuning it.')
    parser.add_argument('--fixed_amp_neg', type=float, default=None, 
                        help='Set a fixed value for amp_neg and skip tuning it.')
    parser.add_argument('--fixed_amp_factor', type=float, default=None, 
                        help='Set a fixed value for amp_factor and skip tuning it.')
    
    return parser.parse_args()


def load_builder(
    build_method: str,
    model_path: str,
    data_path: str,
    example_subset: str = None,
    max_samples: int = 100,
    save_svd: bool = False,
    save_traditional: bool = True,
    chat: bool = False
):
    if build_method == "synth":
        from src.custom_builders.synthetic_qa_builder import SynthQABuilder
        
        builder = SynthQABuilder(
            model_path=model_path,
            data_path=data_path,
            layers="all",
            top_pct=None, # placeholder for tuning
            feature=None,
            max_samples=max_samples,
            min_diff=None, # placeholder for tuning
            chat=chat,
            save_svd=save_svd,
            save_traditional=save_traditional 
        )
    elif build_method == "synth_ada":
        from src.custom_builders.adaptive.synthetic_qa_builder_adaptive import SynthQABuilder
        
        builder = SynthQABuilder(
            model_path=model_path,
            data_path=data_path,
            layers="all",
            top_pct=None, # placeholder for tuning
            feature=None,
            max_samples=max_samples,
            min_diff=None, # placeholder for tuning
            chat=chat,
            save_svd=save_svd,
            save_traditional=save_traditional
        )
    elif build_method == "efficacy" or build_method == "paraphrase":
        from src.custom_builders.adaptive.counterfact_qa_builder import CounterfactQABuilder
        
        builder = CounterfactQABuilder(
            model_path=model_path,
            data_path=data_path,
            example_subset=example_subset,
            benchmark_name=build_method,
            layers="all",
            top_pct=None, # placeholder for tuning
            feature=None,
            max_samples=max_samples,
            min_diff=None, # placeholder for tuning
            chat=chat,
            save_svd=save_svd,
            save_traditional=save_traditional
        )
    elif build_method == "biabios":
        from src.custom_builders.adaptive.biasbios_qa_builder import BiasBiosQABuilder
        
        builder = BiasBiosQABuilder(
            model_path=model_path,
            data_path=data_path,
            example_subset=example_subset,
            layers="all",
            top_pct=None, # placeholder for tuning
            feature=None,
            max_samples=max_samples,
            min_diff=None, # placeholder for tuning
            chat=chat,
            save_svd=save_svd,
            save_traditional=save_traditional
        )
    elif build_method == "hotpotqa":
        from src.custom_builders.adaptive.hotpot_qa_builder import HotpotQABuilder
        
        builder = HotpotQABuilder(
            model_path=model_path,
            data_path=data_path,
            layers="all",
            top_pct=None, # placeholder for tuning
            feature=None,
            max_samples=max_samples,
            min_diff=None, # placeholder for tuning
            chat=chat,
            save_svd=save_svd,
            save_traditional=save_traditional
        )
    else:
        raise ValueError(f"{build_method=} not found")

    return builder


def load_model(
    model_type: str, 
    model_path: str, 
    proj_dir: str,
    use_proj_neg: bool = False,
    top_k_singular: int = 5,
    combination_method: str = "weighted_top_k"
): 
    model_basename = model_path.split("/")[-1].strip()
    if model_type == "seka":
        from src.model.seka_llm import SEKALLM
        
        pos_pt_path = f"{proj_dir}/{model_basename}_pos_proj.pt"
        neg_pt_path = f"{proj_dir}/{model_basename}_neg_proj.pt" if use_proj_neg else None
        print(pos_pt_path)
        print(neg_pt_path)
        
        model = SEKALLM(
            model_path,
            pos_pt=pos_pt_path,
            neg_pt=neg_pt_path,
            marker_start=MARKER_START,
            marker_end=MARKER_END,
            layers=LAYERS,
            amplify_pos=None, # placeholder for tuning
            amplify_neg=None, # placeholder for tuning
            feature_function=None,
            torch_dtype="auto",
            device="auto"
        )
        
    elif model_type == "ada_seka":
        from src.model.adaptive_seka_llm import AdaptiveSEKALLM
        
        if adaptive_expert_path is None:
            raise ValueError("Adaptive SEKA requires an adaptive expert path.")
        expert_path = json.load(open(adaptive_expert_path, "r"))
        
        adaptive_expert_path = f"{proj_dir}/{model_basename}_pos_svd.pt"
        print(adaptive_expert_path)
        
        model = AdaptiveSEKALLM(
            model_path,
            expert_paths=expert_path,
            marker_start=MARKER_START,
            marker_end=MARKER_END,
            layers=LAYERS,
            top_k_singular=top_k_singular,
            combination_method=combination_method,
            amplify_factor=None, # placeholder for tuning
            device="auto",
        )
    elif model_type == "varnorm_seka":
        from src.model.var_seka_llm import VarReduceSEKALLM
        
        pos_pt_path = f"{proj_dir}/{model_basename}_pos_proj.pt"
        pos_lam_path = f"{proj_dir}/{model_basename}_pos_lambda.pt"
        print(pos_pt_path)
        print(pos_lam_path)
        
        model = VarReduceSEKALLM(
            model_path,
            pos_pt=pos_pt_path,
            marker_start=MARKER_START,
            marker_end=MARKER_END,
            layers=LAYERS,
            pos_lam=pos_lam_path,
            feature_function=None,
            torch_dtype="auto",
            device="auto"
        )
    else:
        raise ValueError(f"{model_type=} not found")
    
    return model


def load_task(
    task: str, 
    data_path: str,
    model,
    example_subset: str,
):
    tokenizer = model.tok
    
    if task == "counterfact":
        from benchmarks.counterfact.preprocess import load_dataset
        from benchmarks.counterfact.evaluate import (
            counterfact_efficacy, 
            counterfact_paraphrase,
        )
        
        datasets.disable_caching()
        val_dataset = load_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            attribute_no_entity=False,
            example_subset=example_subset
        )
        test_dataset = load_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            attribute_no_entity=False,
        )
        
        def eval_func(
            model,
            dataset,
            batch_size,
            max_length,
            max_new_tokens,
            chat, 
        ):
            efficacy_score = 0.0
            paraphrase_score = 0.0
            for benchmark_name in ["efficacy", "paraphrase"]:                    
                if benchmark_name == "efficacy":
                    eval_func = counterfact_efficacy  
                else:
                    eval_func = counterfact_paraphrase
                
                results = eval_func(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    batch_size=batch_size,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    add_unmediated_fact=True,
                    chat=chat,
                    seka=True,
                    add_marker=True,
                    marker_start=MARKER_START,
                    marker_end=MARKER_END,
                )
                score = results.metrics.score.mean
                if benchmark_name == "efficacy":
                    efficacy_score = score
                else:
                    paraphrase_score = score
            return efficacy_score * 100, paraphrase_score * 100
        
        opt_directions = ['maximize', 'maximize']
            
    elif task == "biasbios":
        from benchmarks.biasbios.preprocess import load_dataset
        from benchmarks.biasbios.evaluate import biasbios_prediction_evaluation
        
        val_dataset = load_dataset(
            data_path, 
            attribute_no_entity=False,
            example_subset=example_subset
        )
        test_dataset = load_dataset(
            data_path=data_path,
            attribute_no_entity=False,
        )
        
        def eval_func(
            model,
            dataset,
            batch_size,
            max_length,
            max_new_tokens,
            chat, 
        ):
            results = biasbios_prediction_evaluation(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                data_path=data_path,
                batch_size=batch_size,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                desc="BiasBios Evaluation",
                add_marker=True,
                marker_start=MARKER_START,
                marker_end=MARKER_END,
                seka=True,
                chat=chat, 
            )
            acc = results.metrics.top1_accuracy
            return acc * 100

        opt_directions = 'maximize'
        
    elif task == "pronchange":
        from benchmarks.biasbios.preprocess import load_dataset
        from benchmarks.biasbios.evaluate import biasbios_instruction_evaluation, BiosBiasInstructionEvaluationResults
        
        val_dataset = load_dataset(
            data_path=data_path,
            attribute_no_entity=False,
            example_subset=example_subset
        )
        test_dataset = load_dataset(
            data_path=data_path,
            attribute_no_entity=False,
        )
        
        def eval_func(
            model,
            dataset,
            batch_size,
            max_length,
            max_new_tokens,
            chat
        ):
            evluation_result: BiosBiasInstructionEvaluationResults = biasbios_instruction_evaluation(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                data_path=data_path,
                task="pronchange", 
                prompt_idx=3, 
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                desc="Instruction evaluation [LM]",
                add_marker=True,
                marker_start=MARKER_START,
                marker_end=MARKER_END,
                seka=True,
                chat=chat, 
            )
            acc = evluation_result.metrics.instruction_evaluation["pron_sub_acc"]
            all_acc = evluation_result.metrics.instruction_evaluation["pron_all_acc"]
            return acc * 100, all_acc * 100
        
        opt_directions = ['maximize', 'maximize']
    
    else:
        raise ValueError(f"{task=} not found")

    return eval_func, val_dataset, test_dataset, opt_directions


def get_param(trial: optuna.Trial, args, name: str, log: bool = False):
    """Gets a hyperparameter from a fixed arg or suggests it from a trial."""
    fixed_value = getattr(args, f"fixed_{name}", None)
    if fixed_value is not None:
        return fixed_value
    
    param_range = getattr(args, f"{name}_range")
    return trial.suggest_float(name, param_range[0], param_range[1], log=log)


def objective(
    trial: optuna.Trial, 
    args, 
    builder, 
    eval_func,
    model,
    dataset,
):
    # Set proj builder params
    builder.top_pct = get_param(trial, args, "top_pct", log=False)
    builder.min_diff = get_param(trial, args, "min_diff", log=True)
    
    if args.model_type == "seka":
        # Set SEKA params
        model.amplify_pos = get_param(trial, args, "amp_pos", log=False)
        model.amplify_neg = get_param(trial, args, "amp_neg", log=False)
    elif args.model_type == "ada_seka":
        # Set AdaSEKA params
        model.amplify_factor = get_param(trial, args, "amp_factor", log=True)
    elif args.model_type == "varnorm_seka":
        pass
    else:
        raise ValueError
    
    builder.run(args.proj_dir)
    return eval_func(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        chat=args.chat
    )
    

def test(builder, eval_func, model, dataset, best_params, args):
    builder.top_pct = best_params.get("top_pct", args.fixed_top_pct)
    builder.min_diff = best_params.get("min_diff", args.fixed_min_diff)
    
    if args.model_type == "seka":
        # Set SEKA params
        model.amplify_pos = best_params.get("amp_pos", args.fixed_amp_pos)
        model.amplify_neg = best_params.get("amp_neg", args.fixed_amp_neg)
    elif args.model_type == "ada_seka":
        # Set AdaSEKA params
        model.amplify_factor = best_params.get("amp_factor", args.fixed_amp_factor)
    elif args.model_type == "varnorm_seka":
        pass
    else:
        raise ValueError
    
    builder.run(args.proj_dir)
    return eval_func(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        chat=args.chat
    )
    

def main():
    args = parse_args()
    print(args)
    
    builder = load_builder(
        build_method=args.build_method,
        model_path=args.model_path,
        data_path=args.build_data_path,
        example_subset=args.build_example_subset,
        max_samples=args.max_samples,
        save_svd=args.save_svd,
        save_traditional=args.save_traditional,
    )

    model = load_model(
        model_type=args.model_type,
        model_path=args.model_path,
        proj_dir=args.proj_dir,
        use_proj_neg=args.use_proj_neg,
        top_k_singular=args.top_k_singular,
        combination_method=args.combination_method
    )
    eval_func, val_dataset, test_dataset, opt_directions = load_task(
        task=args.task,
        data_path=args.eval_data_path,
        model=model,
        example_subset=args.eval_example_subset
    )
    
    study = optuna.create_study(directions=opt_directions)
    
    # Use a lambda function to pass the pre-loaded objects to the objective
    study.optimize(
        lambda trial: objective(
            trial, 
            args,
            builder=builder, 
            eval_func=eval_func,
            model=model,
            dataset=val_dataset, 
        ), 
        n_trials=args.n_trials
    )
    
    # Get the best hyperparameters
    print("Best trials:")
    for trial in study.best_trials:
        print(f"  Values: {trial.values}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        test_res = test(
            builder=builder,
            eval_func=eval_func,
            model=model,
            dataset=test_dataset,
            best_params=trial.params,
            args=args,
        )
        print("  Test Perf: ")
        print(f"    {test_res}")


if __name__ == "__main__":
    main()