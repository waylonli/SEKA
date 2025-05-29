
import os
import datasets
import logging
from functools import partial

from benchmarks.utils.pasta_utils import ContextMediationInput, prompt_in_context_from_sample
from benchmarks.utils.typing_uils import Dataset
from typing import Any

logger = logging.getLogger(__name__)

def prompt_in_context_from_batch(
    batch: ContextMediationInput,
    output_key: str = "prompt_in_context",
    **kwargs: Any,
) -> dict:
    """Compute prompt in context from batch."""
    is_batched = not isinstance(batch["entity"], str)
    if isinstance(batch["entity"], str):
        entities = [batch["entity"]]
    if isinstance(batch["prompt"], str):
        prompts = [batch["prompt"]]
    if isinstance(batch["context"], str):
        contexts = [batch["context"]]

    prompts_in_context = []
    for entity, prompt, context in zip(entities, prompts, contexts):
        prompt_in_context = prompt_in_context_from_sample(
            entity, prompt, context, **kwargs
        )
        prompts_in_context.append(prompt_in_context)

    return {output_key: prompts_in_context if is_batched else prompts_in_context[0]}

def _load_bias_in_bios(file: str, **kwargs: Any) -> Dataset:
    """Load the Bias in Bios datast, if possible."""
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"biosbias file not found: {file}"
            "\nnote this dataset cannot be downloaded automatically, "
            "so you will have to retrieve it by following the instructions "
            "at the github repo https://github.com/microsoft/biosbias and pass "
            "the .pkl file in via the -f flag"
        )

    dataset = datasets.load_dataset("json", data_files=str(file), **kwargs)

    assert isinstance(
        dataset, datasets.arrow_dataset.Dataset | datasets.dataset_dict.DatasetDict
    ), type(dataset)
    return dataset

def load_dataset(
    data_path: str,
    attribute_no_entity: bool = False,
    example_subset: str = None
):
    if example_subset is not None:
        split = f"train[{example_subset}]"
    else:
        split = "train[5000:10000]"
    dataset = _load_bias_in_bios(file=data_path, split=split)
    
    if attribute_no_entity:
        dataset = dataset.map(
            lambda e: {"context": e["attribute"]}, desc="set context=attribute"
        )
    
    dataset = dataset.map(
        partial(prompt_in_context_from_batch, output_key="prompt", context_suffix="\n\n"),
        desc="precompute prompt in context",
        keep_in_memory=True,
    )
    
    return dataset