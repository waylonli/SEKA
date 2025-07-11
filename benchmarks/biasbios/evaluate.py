import logging
import torch
import datasets
from collections import defaultdict
from tqdm import tqdm
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from anchoring import spa_tokenize, SPALogitsProcessor
from transformers import PreTrainedModel, PreTrainedTokenizer

from benchmarks.biasbios.preprocess import _load_bias_in_bios
from benchmarks.biasbios.evaluator import InstructionEvaluator
from benchmarks.utils.pasta_utils import (
    Metric,
    get_column_names,
    vector_similarity,
    weighted_n_gram_entropy,
    first_token_ids_from_batch
)
from benchmarks.utils.typing_uils import Dataset

from src.model import SEKALLM
from src.utils import encode_with_markers
from pastalib.pasta import PASTA

from typing import cast

logger = logging.getLogger(__name__)

DEFAULT_MAX_LENGTH_ERROR_CORRECTION = 150

@dataclass(frozen=True)
class BiasBiosEvaluationSample:
    """Wrapper around error correction sample."""

    id: str
    prompt: str
    generation: str

    predictions: list[str]
    target: str

    logp_predictions: list[float]
    logp_target: float

    fluency: float
    consistency: float


@dataclass(frozen=True)
class BiasBiosEvaluationMetrics(DataClassJsonMixin):
    """Wrapper around aggregated error correction metrics."""

    top1_accuracy: float
    topk_accuracy: float
    k: int

    fluency: Metric
    consistency: Metric


@dataclass(frozen=True)
class BiasBiosEvaluationResults(DataClassJsonMixin):
    """Wrapper around error correction benchmark."""

    samples: list[BiasBiosEvaluationSample]
    metrics: BiasBiosEvaluationMetrics
    
@dataclass(frozen=True)
class InstructionEvaluationSample:
    """Wrapper around format evaluation sample."""

    id: str
    prompt: str
    generation: str

    predictions: list[str]
    target: str

    logp_predictions: list[float]
    logp_target: float

    fluency: float
    consistency: float

    instruction_evaluation: dict 
    sample_attn_scores: dict 
    
@dataclass(frozen=True)
class InstructionEvaluationMetrics(DataClassJsonMixin):
    """Wrapper around aggregated error correction metrics."""

    top1_accuracy: float
    topk_accuracy: float
    k: int

    fluency: Metric
    consistency: Metric

    instruction_evaluation: dict 
    
@dataclass(frozen=True)
class BiosBiasInstructionEvaluationResults(DataClassJsonMixin):
    """Wrapper around error correction benchmark."""

    samples: list[InstructionEvaluationSample]
    metrics: InstructionEvaluationMetrics 
    attentions: dict 

def load_biasbios_tfidf_vectorizer(data_path: str) -> TfidfVectorizer:
    """Load the tfidf vectorizer for Bias in Bios."""
    logger.info("loading full biasbios dataset for tfidf vectorizer")
    dataset = cast(
        datasets.arrow_dataset.Dataset, _load_bias_in_bios(data_path, split="train")
    )

    texts = [x["source"]["bio"] for x in dataset]
    logger.info(f"create biasbios tfidf vectorizer from {len(texts)} bios")

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(texts)
    return tfidf_vectorizer


@torch.inference_mode()
def biasbios_prediction_evaluation(
    model: PreTrainedModel | SEKALLM,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    data_path: str,
    references: dict | None = None,
    batch_size: int = 16,
    top_k: int = 3,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
    desc: str | None = None,
    add_marker: bool = False,
    marker_start: str | None = None,
    marker_end: str | None = None,
    chat: bool = False,
    seka: bool = False,
    pasta: PASTA | None = None,
    anchor: bool = False,
    anchor_strength: float = 1.6,
) -> BiasBiosEvaluationResults:
    """Run BiasBios prediction benchmark (case-insensitive evaluation)."""

    if max_length is None and max_new_tokens is None:
        max_length = DEFAULT_MAX_LENGTH_ERROR_CORRECTION
    if desc is None:
        desc = "BiasBios Evaluation"

    tfidf_vectorizer = load_biasbios_tfidf_vectorizer(data_path)
    if references is None:
        references = defaultdict(list)
        for sample in dataset:
            references[sample["target_mediated"]].append(sample["source"]["bio"])

    # Canonical labels (lowercase)
    labels = sorted({x["target_mediated"].lower() for x in dataset})
    label_add_space = False if not "gemma" in model.__class__.__name__.lower() else True
    # label_add_space = True

    def all_case_variants(label):
        # Extend as needed
        return [label, label.capitalize(), label.upper()]

    # Map canonical label -> all token ids for its case variants
    label_to_token_ids = {}
    for label in labels:
        variants = all_case_variants(label)
        token_ids = []
        for v in variants:
            try:
                ids = first_token_ids_from_batch(tokenizer, [v], add_space=label_add_space)
                token_ids.append(ids[0])
            except Exception:
                continue
        label_to_token_ids[label] = token_ids

    reference_tfidfs = {
        key: tfidf_vectorizer.transform(texts).mean(axis=0).A
        for key, texts in tqdm(references.items(), desc=f"{desc} [reference tfidfs]")
    }

    columns = get_column_names(dataset, exclude=["target_unmediated"])
    with dataset.formatted_as("torch", columns=columns):
        loader = torch.utils.data.DataLoader(
            cast(torch.utils.data.Dataset, dataset),
            batch_size=batch_size,
        )

        samples = []
        for batch in tqdm(loader, desc=desc):
            ids = batch["id"]
            targets = [t.lower() for t in batch["target_mediated"]]


            if "gemma" in model.__class__.__name__.lower():
                batch["prompt"] = [p.strip() for p in batch["prompt"]]
            elif hasattr(model, "model") and "gemma3" in model.model.__class__.__name__.lower():
                batch["prompt"] = [p.strip() for p in batch["prompt"] if p.strip()]
            
            targets_idx = [label_to_token_ids[t][0] for t in targets]  # Use first variant for target
            
            if add_marker:
                batch['prompt'] = [
                    prompt.replace(attr, marker_start+attr+marker_end) for prompt,attr in zip(batch['prompt'], batch['attribute'])
                ]
            
            prompts = batch['prompt']

            if chat:
                prompts = [tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                ) for prompt in prompts]

            if seka:
                input_ids, steer_mask, attention_mask = encode_with_markers(
                    prompts, tokenizer,
                    marker_start, marker_end)
                input_ids = input_ids.to(model.device)
                steer_mask = steer_mask.to(model.device)
                attention_mask = attention_mask.to(model.device)

                outputs = model.generate(
                    ids=input_ids,
                    steer=True,
                    steer_mask=steer_mask,
                    attention_mask=attention_mask,
                    return_raw=True,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
                generations = tokenizer.batch_decode(
                    outputs.sequences[:, input_ids.shape[1] :],
                    skip_special_tokens=True,
                )
            elif pasta:
                inputs = tokenizer(
                    prompts, return_tensors="pt", 
                    return_offsets_mapping=True,
                    truncation=True, padding=True
                ).to(model.device)
                offset_mapping = inputs.pop("offset_mapping")
                with pasta.apply_steering(
                    model=model, 
                    strings=prompts, 
                    substrings=batch['attribute'], 
                    model_input=inputs, 
                    offsets_mapping=offset_mapping
                ) as steered_model: 
                    outputs = steered_model.generate(**inputs, 
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_length=max_length,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                        output_attentions=True,
                        do_sample=False,
                    )
                generations = tokenizer.batch_decode(
                    outputs.sequences[:, inputs.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )
            elif anchor:
                # Create SPA logits processor
                tokenizer.pad_token_id = tokenizer.eos_token_id
                main_inputs, aux_inputs, mask_token = spa_tokenize(
                    prompt_with_anchors=prompts[0],
                    tokenizer=tokenizer,
                    global_anchors=[],
                    device=model.device
                )
                anchor_processor = SPALogitsProcessor(
                    aux_model=model,
                    aux_input_ids=aux_inputs,
                    strength=anchor_strength,
                    modulated_by_prob=False,
                    use_attention_mask=True,
                    mask_token=mask_token,
                    tokenizer=tokenizer
                )

                # import pdb; pdb.set_trace()
                # Generate text with SPA
                outputs = model.generate(
                    input_ids=main_inputs,
                    attention_mask=torch.ones_like(main_inputs),
                    logits_processor=[anchor_processor],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                generations = tokenizer.batch_decode(
                    outputs.sequences[:, main_inputs.shape[1]:],
                    skip_special_tokens=True,
                )
            else:
                inputs = tokenizer(prompts, return_tensors="pt",
                                truncation=True, padding=True).to(model.device)
                outputs = model.generate(**inputs,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
                generations = tokenizer.batch_decode(
                    outputs.sequences[:, inputs.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )

            distributions = torch.log_softmax(outputs.scores[0], dim=-1)

            for sid, prompt, distribution, generation, target, target_idx in zip(
                ids, prompts, distributions, generations, targets, targets_idx
            ):
                # Aggregate log-probs across all case variants for each label
                agg_label_log_probs = []
                for label in labels:
                    idxs = label_to_token_ids[label]
                    logps = [distribution[idx] for idx in idxs if idx is not None]
                    agg_label_log_probs.append(torch.stack(logps).max() if logps else torch.tensor(float('-inf')))
                agg_label_log_probs = torch.stack(agg_label_log_probs)

                logp_predictions, predictions_idx = agg_label_log_probs.topk(k=top_k, dim=-1)
                predictions = [labels[idx] for idx in predictions_idx]

                # For the target, take the max over case variants as well
                logp_target = agg_label_log_probs[labels.index(target)]

                fluency_score = weighted_n_gram_entropy(generation)

                [generation_tfidf] = tfidf_vectorizer.transform([generation]).A
                reference_tfidf = reference_tfidfs[target]
                consistency_score = vector_similarity(
                    generation_tfidf.squeeze(), reference_tfidf.squeeze()
                )

                # import pdb; pdb.set_trace()
                sample = BiasBiosEvaluationSample(
                    id=sid,
                    prompt=prompt,
                    generation=generation,
                    predictions=predictions,
                    logp_predictions=logp_predictions.tolist(),
                    target=target,
                    logp_target=logp_target.item(),
                    fluency=fluency_score,
                    consistency=consistency_score,
                )
                samples.append(sample)

    n_correct_top1 = sum(x.predictions[0] == x.target for x in samples)
    n_correct_topk = sum(x.target in x.predictions for x in samples)
    top1_accuracy = n_correct_top1 / len(samples)
    topk_accuracy = n_correct_topk / len(samples)

    fluency = Metric.aggregate([x.fluency for x in samples], store_values=False)
    consistency = Metric.aggregate(
        [x.consistency for x in samples], store_values=False
    )

    error_correction_metrics = BiasBiosEvaluationMetrics(
        top1_accuracy=top1_accuracy,
        topk_accuracy=topk_accuracy,
        k=top_k,
        fluency=fluency,
        consistency=consistency,
    )

    return BiasBiosEvaluationResults(
        samples=samples,
        metrics=error_correction_metrics,
    )



@torch.inference_mode()
def biasbios_instruction_evaluation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    data_path: str,
    task: str, 
    prompt_idx: int | list | None = None, 
    references: dict | None = None,
    batch_size: int = 16,
    top_k: int = 3,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
    desc: str | None = None,
    add_marker: bool = False,
    marker_start: str | None = None,
    marker_end: str | None = None,
    chat: bool = False,
    seka: bool = False,
    pasta: PASTA | None = None,
    anchor: bool = False,
    anchor_strength: float = 1.6,
) -> BiosBiasInstructionEvaluationResults:
    """ Evaluate the instruction following tasks  

    This benchmark evaluation model ability of instruction following 

    Args:
        mt: The model to evaluate. 
        dataset: The dataset to evaluate on.
        task: The name of task (json|pronchange). 
        prompt_idx: The index of prompt template. 
        pasta_steerer: The PASTA steerer to steer the model generaion. 
        emphasized_text: The input spans to be highlighted. 
        tfidf_vectorizer: For computing consistency score.
        references: Mapping from label to reference texts for that label. By default,
            full bios for each label will be used.
        batch_size: Batch size for model.
        top_k: Compute top-k labels predicted by model.
        prompt_key: Which column in dataset to use as prompt.
        max_length: Max sequence length (input+output).
        max_new_tokens: Max number of new tokens to generate. Cannot be used with
            `max_length`, see huggingface docs.
        device: Send model and data to this device.
        desc: TQDM description. 
        add_few_shot: Whether to apply few-shot prompting. 
        few_shot_index: Sample index for few shots. 

    Returns:
        Benchmark results.
    """
    if max_length is None and max_new_tokens is None:
        max_length = DEFAULT_MAX_LENGTH_ERROR_CORRECTION
    if desc is None:
        desc = "Instruction Evaluation"

    tfidf_vectorizer = load_biasbios_tfidf_vectorizer(data_path)
    if references is None:
        references = defaultdict(list)
        for sample in dataset:
            references[sample["target_mediated"]].append(sample["source"]["bio"])

    # Load the InstructionEvaluator 
    evaluator = InstructionEvaluator(task, prompt_idx) 

    labels = sorted({x["target_mediated"] for x in dataset})
    label_add_space = False
    labels_token_idx = first_token_ids_from_batch(tokenizer, labels, add_space=label_add_space)

    reference_tfidfs = {
        key: tfidf_vectorizer.transform(texts).mean(axis=0).A
        for key, texts in tqdm(references.items(), desc=f"{desc} [reference tfidfs]")
    }

    columns = get_column_names(dataset, exclude=["target_unmediated"])
    with dataset.formatted_as("torch", columns=columns):
        loader = torch.utils.data.DataLoader(
            cast(torch.utils.data.Dataset, dataset),
            batch_size=batch_size,
        )

        samples = []
        for batch in tqdm(loader, desc=desc):
            ids = batch["id"]
            contexts = batch["context"]
            attributes = batch["attribute"]
            entities = batch["entity"]
            targets = batch["target_mediated"]

            prompts, instructions, entities, (ids, attributes, targets) = evaluator.parapare_prompt_inputs(
                contexts, entities, (ids, attributes, targets)
            )
            targets_idx = first_token_ids_from_batch(tokenizer, targets, add_space=label_add_space)

            if add_marker:
                prompts = [
                    prompt.replace(instruct, marker_start + instruct + marker_end)
                    for prompt, instruct in zip(prompts, instructions)
                ]

            if chat:
                prompts = [tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                ) for prompt in prompts]

            if seka:
                input_ids, steer_mask, attention_mask = encode_with_markers(
                    prompts,
                    tokenizer,
                    marker_start,
                    marker_end)
                input_ids = input_ids.to(model.device)
                steer_mask = steer_mask.to(model.device)
                attention_mask = attention_mask.to(model.device)
                outputs = model.generate(
                    ids=input_ids,
                    steer=True,
                    steer_mask=steer_mask,
                    attention_mask=attention_mask,
                    return_raw=True,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
                generations = tokenizer.batch_decode(
                    outputs.sequences[:, input_ids.shape[1] :],
                    skip_special_tokens=True,
                )
            elif pasta:
                # Prepare PASTA-steered generation (structure matches above for clarity)
                # Tokenize as in the PASTA section of biasbios_prediction_evaluation
                inputs = tokenizer(
                    prompts, return_tensors="pt",
                    return_offsets_mapping=True,
                    truncation=True, padding=True
                ).to(model.device)
                offset_mapping = inputs.pop("offset_mapping")
                with pasta.apply_steering(
                        model=model,
                        strings=prompts,
                        substrings=attributes,
                        model_input=inputs,
                        offsets_mapping=offset_mapping
                ) as steered_model:
                    outputs = steered_model.generate(
                        **inputs,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_length=max_length,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                        output_attentions=True,
                    )
                generations = tokenizer.batch_decode(
                    outputs.sequences[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
            elif anchor:
                # Create SPA logits processor
                main_inputs, aux_inputs, mask_token = spa_tokenize(
                    prompt_with_anchors=prompts[0],
                    tokenizer=tokenizer,
                    global_anchors=[],
                    device=model.device
                )
                anchor_processor = SPALogitsProcessor(
                    aux_model=model,
                    aux_input_ids=aux_inputs,
                    strength=anchor_strength,
                    modulated_by_prob=False,
                    use_attention_mask=True,
                    mask_token=mask_token,
                    tokenizer=tokenizer
                )
                # Generate text with SPA
                outputs = model.generate(
                    input_ids=main_inputs,
                    attention_mask=torch.ones_like(main_inputs),
                    logits_processor=[anchor_processor],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                generations = tokenizer.batch_decode(
                    outputs.sequences[:, main_inputs.shape[1]:],
                    skip_special_tokens=True,
                )
            else:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                inputs = tokenizer(prompts, return_tensors="pt",
                                truncation=True, padding=True,
                                ).to(model.device)
                outputs = model.generate(**inputs,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
                generations = tokenizer.batch_decode(
                    outputs.sequences[:, inputs.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )

            distributions = torch.log_softmax(outputs.scores[0], dim=-1)

            for idx, sid, prompt, distribution, generation, target, target_idx in zip(
               range(len(ids)), ids, prompts, distributions, generations, targets, targets_idx
            ):
                label_log_probs = distribution[labels_token_idx]

                logp_predictions, predictions_idx = label_log_probs.topk(
                    k=top_k, dim=-1
                )
                predictions = [labels[idx] for idx in predictions_idx]

                logp_target = distribution[target_idx]

                fluency_score = weighted_n_gram_entropy(generation)

                [generation_tfidf] = tfidf_vectorizer.transform([generation]).A
                reference_tfidf = reference_tfidfs[target]
                consistency_score = vector_similarity(
                    generation_tfidf.squeeze(), reference_tfidf.squeeze()
                )

                instruction_evaluation = evaluator.evaluate_sample(generation=generation, target=target)

                sample = InstructionEvaluationSample(
                    id=sid,
                    prompt=prompt,
                    generation=generation,
                    predictions=predictions,
                    logp_predictions=logp_predictions.tolist(),
                    target=target,
                    logp_target=logp_target.item(),
                    fluency=fluency_score,
                    consistency=consistency_score,
                    instruction_evaluation=instruction_evaluation, 
                    sample_attn_scores=None, 
                )

                samples.append(sample)

    n_correct_top1 = sum(x.predictions[0] == x.target for x in samples)
    n_correct_topk = sum(x.target in x.predictions for x in samples)
    top1_accuracy = n_correct_top1 / len(samples)
    topk_accuracy = n_correct_topk / len(samples)

    fluency = Metric.aggregate([x.fluency for x in samples], store_values=False)
    consistency = Metric.aggregate(
        [x.consistency for x in samples], store_values=False
    )

    instruction_evaluation = evaluator.aggregate_evaluation_results(samples)
    instruction_evaluation_metrics = InstructionEvaluationMetrics(
        top1_accuracy=top1_accuracy,
        topk_accuracy=topk_accuracy,
        k=top_k,
        fluency=fluency,
        consistency=consistency,
        instruction_evaluation=instruction_evaluation, 
    )

    return BiosBiasInstructionEvaluationResults(
        samples=samples,
        metrics=instruction_evaluation_metrics,
        attentions=None, 
    )