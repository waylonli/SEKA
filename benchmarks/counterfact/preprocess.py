import torch
import logging
import datasets
from functools import partial
from transformers import PreTrainedTokenizer

from benchmarks.utils.typing_uils import Dataset
from benchmarks.utils.pasta_utils import (
    ContextMediationInput, 
    prompt_in_context_from_sample,
    first_token_ids_from_batch
)

from typing import Any

logger = logging.getLogger(__name__)

def _load_counterfact(
    file: str,
    **kwargs: Any,
) -> Dataset:
    """Download and format the counterfact dataset."""
    dataset = datasets.load_dataset("json", data_files=str(file), **kwargs)
    assert isinstance(
        dataset, datasets.arrow_dataset.Dataset | datasets.dataset_dict.DatasetDict
    ), type(dataset)

    return dataset

def _prefix_context(sample: dict) -> dict:
    """Prepend context to all prompts used in the eval."""
    entity = sample["entity"]
    prompt = sample["prompt"]
    context = sample["context"]

    prompt_in_context = prompt_in_context_from_sample(
        entity, prompt, context
    )

    source = {**sample["source"]}
    for key in ("generation_prompts", "paraphrase_prompts"):
        source[key] = [
            prompt_in_context_from_sample(entity, other_prompt, context)
            for other_prompt in source[key]
        ]
        
    return {"source": source, "prompt": prompt_in_context}

def _as_fp32(data: dict) -> dict:
    """Cast all top-level float tensor values to float32."""
    return {
        key: value.float()
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point
        else value
        for key, value in data.items()
    }

def precompute_token_ids(
    sample: ContextMediationInput,
    tokenizer: PreTrainedTokenizer,
    target_token_first_space: bool = False, 
    fp32: bool = False,
):
    precomputed = {}
    for target_key in ("target_mediated", "target_unmediated"):
        target = sample.get(target_key)
        if target is None or any(t is None for t in target):
            continue
        precomputed[f"{target_key}.token_id"] = first_token_ids_from_batch(
            tokenizer, target, target_token_first_space 
        )
        
    if fp32:
        precomputed = _as_fp32(precomputed)

    return precomputed

def load_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    attribute_no_entity: bool = False,
    example_subset: str = None
):
    # Set up the evaluation data 
    logger.info("loading several data sources")
    if example_subset is not None:
        split = f"train[{example_subset}]"
    else:
        split = "train[5000:10000]"
    dataset = _load_counterfact(data_path, split=split)
    if attribute_no_entity:
        dataset = dataset.map(
            lambda e: {"context": e["attribute"]}, desc="set context=attribute"
        )
    
    dataset = dataset.map(
        partial(precompute_token_ids,
            tokenizer=tokenizer,
            fp32=True,
            target_token_first_space=False,
        ),
        desc="precompute target token ids",
        batched=True,
        batch_size=64,
        keep_in_memory=True,
        num_proc=1,
    )
    dataset = dataset.map(_prefix_context, desc="prefix context")
    return dataset
