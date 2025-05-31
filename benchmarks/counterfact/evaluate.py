import logging
import json
import torch
import scipy
import numpy as np
from functools import partial
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from collections import OrderedDict, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import PreTrainedModel, PreTrainedTokenizer

from typing import Sequence, cast

from benchmarks.utils.typing_uils import Dataset, ArrayLike, StrSequence
from benchmarks.counterfact.preprocess import precompute_token_ids
from benchmarks.utils.pasta_utils import (
    ContextMediationSample,
    Metric, 
    get_column_names, 
    _validate_same_length,
    vector_similarity,
    weighted_n_gram_entropy
)

from src.model import SEKALLM
from pastalib.pasta import PASTA

logger = logging.getLogger(__name__)

DEFAULT_MAX_LENGTH = 300
DEFAULT_MAX_LENGTH_ERROR_CORRECTION = 150

AttributeSnippets = dict[str, dict[str, list[dict]]]

@dataclass(frozen=True)
class EfficacySample(DataClassJsonMixin):
    """Wrapper around a single efficacy sample."""

    id: str
    prompt: str
    target_score: float
    comparator_score: float
    
@dataclass(frozen=True)
class EfficacyMetrics(DataClassJsonMixin):
    """Efficacy metrics."""

    score: Metric
    magnitude: Metric

    def without_values(self) -> "EfficacyMetrics":
        """Return the metrics without the values stored."""
        return EfficacyMetrics(
            score=self.score.without_values(),
            magnitude=self.magnitude.without_values(),
        )
    
@dataclass(frozen=True)
class EfficacyBenchmarkResults(DataClassJsonMixin):
    """Wrapper around efficacy benchmark results."""

    samples: list[EfficacySample]
    metrics: EfficacyMetrics

@dataclass(frozen=True)
class CounterFactEvaluationResult(DataClassJsonMixin):
    """Result for single sample from `Editor.evaluate`."""

    sample: dict

    top_tokens: list[str] | None = None
    top_logps: list[float] | None = None
    generations: list[str] | None = None

    target_mediated_score: float | None = None
    target_unmediated_score: float | None = None

@dataclass(frozen=True)
class CounterFactEvaluateRun(DataClassJsonMixin):
    """Wrapper around a list of individual evaluation results."""

    results: list[CounterFactEvaluationResult]
    
@dataclass(frozen=True)
class ParaphraseSample(DataClassJsonMixin):
    """Wrapper around a single paraphrase benchmark sample."""

    id: str
    prompts: list[EfficacySample]
    efficacy_score: float
    efficacy_magnitude: float

@dataclass(frozen=True)
class CounterFactParaphraseBenchmarkResults(DataClassJsonMixin):
    """Wrapper around paraphrase benchmark results."""

    samples: list[ParaphraseSample]
    metrics: EfficacyMetrics
    
@dataclass(frozen=True)
class GenerationSample(DataClassJsonMixin):
    """Wrapper around a single sample from the generation benchmark."""

    id: str
    generations: list[str]
    references: list[str]
    fluency_score: float
    consistency_score: float

@dataclass(frozen=True)
class GenerationMetrics(DataClassJsonMixin):
    """Wrapper around all generation metrics."""

    fluency: Metric
    consistency: Metric
    
@dataclass(frozen=True)
class CounterFactGenerationBenchmarkResults(DataClassJsonMixin):
    """Wrapper around generation benchmark results."""

    samples: list[GenerationSample]
    metrics: GenerationMetrics

@torch.inference_mode()
def counterfact_evaluate(
    model: PreTrainedModel | SEKALLM,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int,
    n_top: int = 3,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
    desc: str | None = None,
    return_mediated: bool = True,
    return_unmediated: bool = True,
    add_unmediated_fact: bool = True,
    add_marker: bool = False,
    marker_start: str | None = None,
    marker_end: str | None = None,
    chat: bool = False,
    seka: bool = False,
    pasta: PASTA | None = None,
) -> CounterFactEvaluateRun:
    include_target_probs = "target_mediated" in dataset.column_names

    if desc is None:
        desc = f"Evaluate CounterFact"
        
    if max_length is None and max_new_tokens is None:
        max_length = DEFAULT_MAX_LENGTH
    
    exclude_columns = []
    if not return_mediated:
        exclude_columns.append("target_mediated")
    if not return_unmediated:
        exclude_columns.append("target_unmediated")
    columns = get_column_names(dataset, exclude=exclude_columns)
    
    results = []
    with dataset.formatted_as("torch", columns=columns):
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        for batch in tqdm(loader, desc=desc):
            prompts = batch["prompt"]
            prompts = batch["prompt"]
            contexts = batch["context"]
            attributes = batch["attribute"]
            targets_mediated = batch["target_mediated"]
            targets_unmediated = batch["target_unmediated"]
            current_batch_size = len(prompts)
            
            if add_unmediated_fact:
                # Modify the prompts and provide the model both new and old facts 
                new_prompts = []
                for prompt, context, target_mediated, target_unmediated in zip(
                    prompts, contexts, targets_mediated, targets_unmediated
                ):
                    unmediated_prefix = "Previously "
                    mediated_prefix = "Currently "
                    unmediated_fact = context.replace(target_mediated, target_unmediated)+". "
                    new_prompt = f"{unmediated_prefix}{unmediated_fact}{mediated_prefix}{prompt}"

                    new_prompts.append(new_prompt)
                prompts = new_prompts
                
            if add_marker:
                prompts = [
                    prompt.replace(attr, marker_start+attr+marker_end) 
                    for prompt, attr in zip(prompts, attributes)
                ]
            
            if chat:
                prompts = [tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                ) for prompt in prompts]
            
            # with open("counterfact_prompts.json", 'w') as f:
            #     json.dump(prompts, f, indent=4)
            #     assert False
            
            if seka:
                outputs = model.generate(
                    ids=prompts,
                    steer=True,
                    return_raw=True,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    temperature=None,
                    top_k=None,
                    top_p=None,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
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
                    substrings=attributes, 
                    model_input=inputs, 
                    offsets_mapping=offset_mapping
                ) as steered_model: 
                    outputs = steered_model.generate(**inputs, 
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True,
                        temperature=None,
                        top_k=None,
                        top_p=None,
                        max_length=max_length,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                        output_attentions=True,
                    )
            else:
                inputs = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True).to(model.device)
                outputs = model.generate(**inputs, 
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    temperature=None,
                    top_k=None,
                    top_p=None,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
            batched_results: dict = {}
            first_token_logps = torch.log_softmax(outputs.scores[0], dim=-1)
            top_logps, top_token_ids = first_token_logps.topk(k=n_top, dim=-1)
            top_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in top_token_ids]
            generations = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

            batched_results[f"top_logps"] = top_logps.tolist()
            batched_results[f"top_tokens"] = top_tokens
            batched_results[f"generations"] = [[g] for g in generations]
            
            if include_target_probs:
                target_keys = []
                if return_mediated:
                    target_keys.append("mediated")
                if return_unmediated:
                    target_keys.append("unmediated")
                # batch_indices = torch.arange(current_batch_size)
                for target_key in target_keys:
                    target_id = batch[f"target_{target_key}.token_id"].to(model.device)
                    target_probs = first_token_logps.gather(dim=1, index=target_id) # shape: (3, batch_size)
                    max_logps, _ = target_probs.max(dim=1) # take max across the 3 variants, per batch‐item
                    batched_results[f"target_{target_key}_score"] = max_logps.tolist()
                    # target_id = batch[f"target_{target_key}.token_id"]
                    # target_probs = first_token_logps[batch_indices, target_id]
                    # target_prob_key = f"target_{target_key}_score"
                    # batched_results[target_prob_key] = target_probs.tolist()

            for bi in range(current_batch_size):
                result: dict = {k: vs[bi] for k, vs in batched_results.items()}
                results.append(result)
            
    # Finally, decorate results with original sample data.
    assert len(results) == len(dataset)
    for sample, result in zip(dataset, results):
        result.update(
            sample={
                key: sample[key]
                for key in ContextMediationSample.__required_keys__
            }
        )

    return CounterFactEvaluateRun([CounterFactEvaluationResult(**r) for r in results])
        
def efficacy(
    p_targets: Sequence[ArrayLike],
    p_comparators: Sequence[ArrayLike],
    assume_log_probs: bool = True,
    store_values: bool = True,
) -> EfficacyMetrics:
    """Compute efficacy on metrics.

    Efficacy is determined by a score and a magnitude. The score is how often
    p(target) > p(comparator). The magnitude is the average p(target) - p(comparator).

    Inputs are two sequences. Each element should be one or more measurements of
    the probability (for e.g. different prompts). This function will first average
    across those inner lists, then average across the whole list.
    """
    _validate_same_length(p_targets=p_targets, p_comparators=p_comparators)

    scores, magnitudes = [], []
    for i, (p_target, p_comparator) in enumerate(zip(p_targets, p_comparators)):
        _validate_same_length(
            **{f"p_target_{i}": p_target, f"p_comparator_{i}": p_comparator}
        )

        if assume_log_probs:
            p_target = np.exp(p_target)
            p_comparator = np.exp(p_comparator)
        else:
            p_target = np.array(p_target)
            p_comparator = np.array(p_comparator)

        score = np.mean(p_target > p_comparator)
        scores.append(score)

        magnitude = np.mean(p_target - p_comparator)
        magnitudes.append(magnitude)

    return EfficacyMetrics(
        score=Metric.aggregate(scores, store_values=store_values),
        magnitude=Metric.aggregate(magnitudes, store_values=store_values),
    )

def counterfact_efficacy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int,
    desc: str | None = None,
    n_top: int = 3,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
    return_mediated: bool = True,
    return_unmediated: bool = True,
    add_unmediated_fact: bool = True,
    add_marker: bool = False, 
    marker_start: str | None = None,
    marker_end: str | None = None,
    chat: bool = False,
    seka: bool = False,
    pasta: PASTA | None = None,
):
    if desc is None:
        desc = "efficacy benchmark"
        
    # Overwrite max_new_tokens
    max_new_tokens = 1
    
    run = counterfact_evaluate(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=batch_size,
        n_top=n_top,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        desc=desc,
        return_mediated=return_mediated,
        return_unmediated=return_unmediated,
        add_unmediated_fact=add_unmediated_fact,
        add_marker=add_marker,
        marker_start=marker_start,
        marker_end=marker_end,
        chat=chat,
        seka=seka,
        pasta=pasta
    )
    
    target_score_key = "target_mediated_score"
    comparator_score_key = "target_unmediated_score"
    
    samples = []
    for result in run.results:
        sid = result.sample["id"]
        prompt = result.sample["prompt"]

        target_score = getattr(result, target_score_key)
        assert target_score is not None

        comparator_score = getattr(result, comparator_score_key)
        assert comparator_score is not None

        logger.debug(f"ID={sid} SCORE_T={target_score} SCORE_COMP={comparator_score}")
        sample = EfficacySample(
            id=sid,
            prompt=prompt,
            target_score=target_score,
            comparator_score=comparator_score,
        )
        samples.append(sample)

    efficacy_metrics = efficacy(
        [[sample.target_score] for sample in samples],
        [[sample.comparator_score] for sample in samples],
        store_values=False,
    )
    return EfficacyBenchmarkResults(samples=samples, metrics=efficacy_metrics)
    
def _counterfact_select_and_flatten(
    dataset: Dataset, column: str, desc: str | None = None
) -> Dataset:
    """Select the given column in counterfact, dedupe it, and flatten it."""
    column_names = get_column_names(dataset)

    def select_and_flatten_counterfact_row(row: dict) -> dict:
        prompts = list(set(row["source"][0][column]))
        result = {"prompt": prompts}
        for key in ContextMediationSample.__required_keys__:
            if key not in result:
                result[key] = [row[key][0]] * len(prompts)
        return result

    return dataset.map(
        select_and_flatten_counterfact_row,
        batched=True,
        batch_size=1,
        remove_columns=column_names,
        desc=desc,
    )

def counterfact_paraphrase(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int,
    desc: str | None = None,
    n_top: int = 3,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
    return_mediated: bool = True,
    return_unmediated: bool = True,
    add_unmediated_fact: bool = True,
    add_marker: bool = False, 
    marker_start: str | None = None,
    marker_end: str | None = None,
    chat: bool = False,
    seka: bool = False,
    pasta: PASTA | None = None,
):
    """Run the CounterFact paraphrase benchmark.

    Since this benchmark relies on extra data, it can only be used with the CounterFact
    dataset. The `counterfact_generation` benchmark is like this as well.

    This function expects that each sample in the dataset supports an access like:

        prompts = sample["source"]["generation_prompts"]

    """
    if desc is None:
        desc = "paraphrase benchmark"
    dataset = _counterfact_select_and_flatten(
        dataset, "paraphrase_prompts", desc=f"{desc} [flatten dataset]"
    )
    dataset = dataset.map(
        partial(precompute_token_ids,
            tokenizer=tokenizer,
            target_token_first_space=False,
        ),
        desc=f"{desc} [recompute token ids]",
        batched=True,
        batch_size=64,
        keep_in_memory=True,
        num_proc=1,
    )
    
    efficacy_benchmark = counterfact_efficacy(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=batch_size,
        n_top=n_top,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        desc=desc,
        return_mediated=return_mediated,
        return_unmediated=return_unmediated,
        add_unmediated_fact=add_unmediated_fact,
        add_marker=add_marker,
        marker_start=marker_start,
        marker_end=marker_end,
        chat=chat,
        seka=seka,
        pasta=pasta,
    )

    results_by_sample_id: dict = defaultdict(list)
    for result in efficacy_benchmark.samples:
        results_by_sample_id[result.id].append(result)
    results_by_sample_id = OrderedDict(results_by_sample_id)

    efficacy_metrics = efficacy(
        [
            [result.target_score for result in results]
            for results in results_by_sample_id.values()
        ],
        [
            [result.comparator_score for result in results]
            for results in results_by_sample_id.values()
        ],
    )

    # Reformat EfficacySample -> ParaphraseSample
    samples = []
    for (sid, results), efficacy_score, efficacy_magnitude in zip(
        results_by_sample_id.items(),
        cast(list, efficacy_metrics.score.values),
        cast(list, efficacy_metrics.magnitude.values),
    ):
        sample = ParaphraseSample(
            id=sid,
            prompts=results,
            efficacy_score=efficacy_score,
            efficacy_magnitude=efficacy_magnitude,
        )
        samples.append(sample)

    return CounterFactParaphraseBenchmarkResults(
        samples=samples,
        metrics=efficacy_metrics.without_values(),
    )
    
def load_attribute_snippets(
    file: str | None = None
) -> AttributeSnippets:
    """Load attribute snippets for different Wikipedia relations/entities.

    This dataset is taken directly from the ROME evaluation. It is not loaded from
    `load_dataset` because it is a mapping, not a sequence. Specifically, it is a
    mapping from Wikidata relation IDs and entity IDs to Wikipedia articles about
    those entities, where the article includes text relevant to the relation. This is
    used to measure consistency of generations with other plausible facts about an
    entity satisfying some relation.

    Args:
        file: Look for attribute snippets at this file. Downloaded otherwise.
        url: Download snippets from this URL.
        overwrite: Overwrite an existing download.

    Returns:
        Mapping from relation ID and entity ID to Wikipedia text.

    """
    with open(file, "r") as f:
        snippets_list = json.load(f)

    attribute_snippets: AttributeSnippets = defaultdict(lambda: defaultdict(list))
    for snippets in snippets_list:
        relation_id = snippets["relation_id"]
        target_id = snippets["target_id"]
        for sample in snippets["samples"]:
            attribute_snippets[relation_id][target_id].append(sample)

    return attribute_snippets

def load_counterfact_tfidf_vectorizer(
    idf_file: str | None = None,
    vocab_file: Path | None = None,
) -> TfidfVectorizer:
    """Load precomputed TF-IDF statistics."""
    idf = np.load(str(idf_file))
    with open(vocab_file, "r") as f:
        vocab = json.load(f)

    # Hack borrowed from ROME:
    # https://github.com/kmeng01/rome/blob/0874014cd9837e4365f3e6f3c71400ef11509e04/dsets/tfidf_stats.py#L17
    class ModifiedTfidfVectorizer(TfidfVectorizer):
        TfidfVectorizer.idf_ = idf

    vec = ModifiedTfidfVectorizer()
    vec.vocabulary_ = vocab
    vec._tfidf._idf_diag = scipy.sparse.spdiags(idf, diags=0, m=len(idf), n=len(idf))
    return vec
    
def _group_results_by_id(results: CounterFactEvaluateRun) -> OrderedDict:
    """Group results by sample ID."""
    grouped = defaultdict(list)
    for result in results.results:
        grouped[result.sample["id"]].append(result)
    return OrderedDict(grouped)

def tfidf_similarity(
    source: str | StrSequence,
    reference: str | StrSequence,
    tfidf_vectorizer: TfidfVectorizer,
) -> float:
    """Return TfIdf similarity between the texts."""
    if isinstance(source, str):
        source = [source]
    if isinstance(reference, str):
        reference = [reference]
    sv, rv = tfidf_vectorizer.transform([" ".join(source), " ".join(reference)]).A
    return vector_similarity(sv, rv)
    
def counterfact_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int,
    attribute_snippets: AttributeSnippets,
    tfidf_vectorizer: TfidfVectorizer,
    desc: str | None = None,
    n_top: int = 3,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
    return_mediated: bool = True,
    return_unmediated: bool = True,
    add_unmediated_fact: bool = True,
    add_marker: bool = False, 
    marker_start: str | None = None,
    marker_end: str | None = None,
    chat: bool = False,
    seka: bool = False,
    pasta: PASTA | None = None,
) -> CounterFactGenerationBenchmarkResults:
    """Run the CounterFact generation benchmark.

    Free-form generates on several "generation prompts" per sample, and records
    the fluency of the generations (measured by weighted n-gram entropy) and
    consistency with other texts about entities with the same attribute.

    This benchmark *requires* the dataset to be CounterFact or something that looks
    like it, since it uses extra data that is specific to CounterFact.

    Specifically, it expects each sample can be accessed like:

        prompts = sample["source"]["generation_prompts"]

    """
    if desc is None:
        desc = "generate benchmark"
    if max_new_tokens is None and max_length is None:
        max_length = DEFAULT_MAX_LENGTH

    dataset = _counterfact_select_and_flatten(
        dataset, "generation_prompts", desc=f"{desc} [flatten dataset]"
    )
    dataset = dataset.map(
        partial(precompute_token_ids,
            tokenizer=tokenizer,
            target_token_first_space=False,
        ),
        desc=f"{desc} [recompute token ids]",
        batched=True,
        batch_size=64,
        keep_in_memory=True,
        num_proc=1,
    )

    run = counterfact_evaluate(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=batch_size,
        n_top=n_top,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        desc=desc,
        return_mediated=return_mediated,
        return_unmediated=return_unmediated,
        add_unmediated_fact=add_unmediated_fact,
        add_marker=add_marker,
        marker_start=marker_start,
        marker_end=marker_end,
        chat=chat,
        seka=seka,
        pasta=pasta,
    )
    generations_key = "generations"
    run_results_by_id = _group_results_by_id(run)

    samples = []
    for sid, results in tqdm(run_results_by_id.items(), desc=f"{desc} [tfidf]"):
        result = next(iter(results))
        cf_requested_rewrite = result.sample["source"]["requested_rewrite"]
        relation_id = cf_requested_rewrite["relation_id"]
        target_id = cf_requested_rewrite["target_new"]["id"]

        generations = [getattr(result, generations_key)[0] for result in results]
        references = [
            snippet["text"] for snippet in attribute_snippets[relation_id][target_id]
        ]

        consistency_score = tfidf_similarity(
            generations, references, tfidf_vectorizer
        )
        fluency_score = weighted_n_gram_entropy(generations)

        entity = result.sample["entity"]
        attribute = result.sample["attribute"]
        logger.debug(f"ID={sid} ENTITY={entity}, ATTR={attribute}")
        logger.debug(f"ID={sid} REFERENCES={references}")
        logger.debug(f"ID={sid} GENERATIONS={generations}")

        sample = GenerationSample(
            id=sid,
            generations=generations,
            references=references,
            fluency_score=fluency_score,
            consistency_score=consistency_score,
        )
        samples.append(sample)

    fluency = Metric.aggregate(
        [sample.fluency_score for sample in samples], store_values=False
    )
    consistency = Metric.aggregate(
        [sample.consistency_score for sample in samples], store_values=False
    )
    generation_metrics = GenerationMetrics(fluency=fluency, consistency=consistency)
    return CounterFactGenerationBenchmarkResults(
        samples=samples, metrics=generation_metrics
    )