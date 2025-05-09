import sys
import nltk
import torch
import logging
import numpy as np
import datasets
from itertools import chain
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from transformers import PreTrainedTokenizer

from typing import TypedDict, Sequence

from utils.typing_uils import StrSequence, ArrayLike, Dataset

def setup_logger():
    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )

@dataclass(frozen=True)
class Metric(DataClassJsonMixin):
    """An aggregate metric."""

    mean: float
    std: float
    values: ArrayLike | None = None

    def without_values(self) -> "Metric":
        """Return the metric without the values stored."""
        return Metric(mean=self.mean, std=self.std)

    @staticmethod
    def aggregate(values: ArrayLike, store_values: bool = True) -> "Metric":
        """Aggregate mean/std of the values."""
        return Metric(
            np.mean(values), np.std(values), values=values if store_values else None
        )

class ContextMediationSample(TypedDict):
    """Single sample that can be used for context mediation analysis."""
    id: str  # Identifier
    entity: str  # "Barack Obama"
    attribute: str  # "invented the iPhone"
    context: str  # "Everyone knows that Barack Obama invented the iPhone."
    prompt: str  # "Barack Obama received a degree in"
    target_mediated: str | None  # "computer science" or not set for generation
    target_unmediated: str | None  # "law" or not set for generation
    source: dict | None  # Where this sample was derived from, e.g. counterfact sample.
    
class ContextMediationBatch(TypedDict):
    """Batch of context mediation samples."""

    id: StrSequence
    entity: StrSequence
    attribute: StrSequence
    context: StrSequence
    prompt: StrSequence
    target_mediated: StrSequence | None
    target_unmediated: StrSequence | None
    source: Sequence[dict] | None

ContextMediationInput = ContextMediationSample | ContextMediationBatch

def get_column_names(dataset: Dataset, exclude: StrSequence | None = None) -> list[str]:
    """Get all column names for the dataset."""
    if isinstance(dataset, datasets.arrow_dataset.Dataset):
        column_names = dataset.column_names
    else:
        assert isinstance(dataset, datasets.dataset_dict.DatasetDict), type(dataset)
        column_names = list(set(chain(*dataset.column_names.values())))

    if exclude is not None:
        column_names = list(set(column_names) - set(exclude))

    return column_names
    
def prompt_in_context_from_sample(
    entity: str,
    prompt: str,
    context: str,
    context_prefix: str | None = None,
    context_suffix: str | None = None,
    prompt_prefix: str | None = None,
) -> str:
    """Compute prompt in context for the sample.

    The prompt in context is simply the "prompt" field in each sample prepended with
    the "context" field. This function tries to make the casing look sensible while
    also not changing the casing of any entity mention.

    Can optionally include prefixes for all contexts and/or for all prompts. This is
    useful for adding function or transition words between the prompt and context so
    that the language model can better reconcile the task.

    Args:
        entity: The entity.
        prompt: The prompt.
        context: The context.
        context_prefix: Prepend this to context.
        context_suffix: Append this to context.
        prompt_prefix: Prepend this to prompt, but after context.


    Returns:
        A single string with the context followed by the prompt.

    """
    if prompt_prefix is not None:
        if not prompt.startswith(entity):
            prompt = prompt[0].lower() + prompt[1:]
        prompt = f"{prompt_prefix}{prompt}"

    if context_prefix is not None:
        if not context.startswith(entity):
            context = context[0].lower() + context[1:]
        context = f"{context_prefix} {context}"

    # Always make sure context is a complete, period-ended sentence.
    context = context.rstrip(". ") + "."

    if context_suffix is not None:
        context = f"{context}{context_suffix}"
    else:
        context += " "

    prompt_in_context = f"{context}{prompt}"
    return prompt_in_context

def _validate_same_length(**kwargs: Sequence | ArrayLike) -> None:
    """Validate all batch sizes are the same."""
    lengths = {key: len(seq) for key, seq in kwargs.items()}
    if len(set(lengths.values())) > 1:
        message = f"inconsistent batch sizes:" + "\n\t"
        message += "\n\t".join(f"{key}={length}" for key, length in lengths.items())
        raise ValueError(message)

def _n_gram_counts(text: str, n: int) -> dict[tuple[str, ...], int]:
    """Return the n-gram counts for the text."""
    tokens = nltk.word_tokenize(text)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)    

def n_gram_entropy(text: str, n: int) -> float:
    """Return entropy of n-gram distribution in text."""
    counts = _n_gram_counts(text, n)
    dist = np.array([count for _, count in counts.items()], dtype=np.float32)
    dist /= dist.sum()
    entropy = np.sum(-dist * np.log(dist) / np.log(2))
    return entropy.item()

def weighted_n_gram_entropy(
    texts: str | StrSequence,
    ns: Sequence[int] = (2, 3),
    weights: Sequence[float] = (2 / 3, 4 / 3),
) -> float:
    """Return weighted n-gram entropy for different values of n."""
    _validate_same_length(ns=ns, weights=weights)
    if isinstance(texts, str):
        texts = [texts]
    entropies = []
    for text in texts:
        entropies_by_n = np.array([n_gram_entropy(text, n) for n in ns])
        entropy = np.mean(entropies_by_n * np.array(weights))
        entropies.append(entropy)
    return np.mean(entropies)

def vector_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) + eps) / (np.linalg.norm(b) + eps)

def first_token_ids_from_batch(
    tokenizer: PreTrainedTokenizer, 
    words: str | StrSequence, 
    add_space: bool = False,
) -> torch.Tensor:
    """Return shape (batch_size,) int tensor with first token ID for each word."""
    # TODO(evandez): Centralize this spacing nonsense.
    if isinstance(words, str):
        words = [words]

    if add_space:
        token_ids = tokenizer([" " + word for word in words], add_special_tokens=False)
    else:
        token_ids = tokenizer(words, add_special_tokens=False)

    return torch.tensor([ti[0] for ti in token_ids.input_ids])
