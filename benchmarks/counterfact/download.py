import os
import json
import wget
import spacy
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

from benchmarks.utils.pasta_utils import ContextMediationSample, setup_logger

logger = logging.getLogger(__name__)

ROME_BASE_URL = "https://rome.baulab.info/data/dsets"
COUNTERFACT_URL = f"{ROME_BASE_URL}/counterfact.json"
ATTRIBUTE_SNIPPETS_URL = f"{ROME_BASE_URL}/attribute_snippets.json"
TFIDF_IDF_URL = f"{ROME_BASE_URL}/idf.npy"
TFIDF_VOCAB_URL = f"{ROME_BASE_URL}/tfidf_vocab.json"

_COUNTERFACT_PARAPHRASE_PROMPT_ARTIFACTS = (" (b. ", "(tr. ", "(min. ")

nlp = spacy.load("en_core_web_sm")

def _rejoin_sents_on_entity(entity: str, sents: list[str]) -> list[str]:
    """Rejoin any splits where the entity was broken into multiple sentences."""
    candidates = [
        index
        for index, (left, right) in enumerate(zip(sents, sents[1:]))
        if (entity not in left and entity not in right and entity in f"{left} {right}")
        or left.endswith(entity)
    ]
    if not candidates:
        return sents

    assert len(candidates) == 1
    [index] = candidates
    merged = f"{sents[index]} {sents[index + 1]}"
    return [*sents[:index], merged, *sents[index + 2 :]]
    
def _strip_counterfact_paraphrase_prompt(entity: str, prompt: str) -> str:
    """Strip the cruft out of one of CounterFact's paraphrase prompts."""
    # Sometimes the paraphrase model stuck in a bunch of newlines, always
    # take what comes after them.
    prompt = prompt.split("\n")[-1].strip()

    # Similarly, it sometimes adds a wikipedia title. In these cases, a good
    # heuristic is just to only take everything beyond and including the entity.
    if prompt.startswith("Category:"):
        prompt = entity + prompt.split(entity)[-1]

    # Another common, weird artifact to get rid of:
    for artifact in _COUNTERFACT_PARAPHRASE_PROMPT_ARTIFACTS:
        if artifact in prompt:
            prompt = prompt.split(artifact)[-1]

    sents = [str(sent) for sent in nlp(prompt).sents]
    sents = _rejoin_sents_on_entity(entity, sents)
    if len(sents) <= 2:
        if entity not in sents[0]:
            assert len(sents) == 2
            sents = [sents[1]]
    else:
        if "?" in sents[-2]:
            sents = [sents[-2], sents[-1]]
        else:
            sents = [sents[-1]]

    stripped = " ".join(sents).strip()

    # If there's still a period in the sentence, but the entity does not have it,
    # it usually means we failed to split a sentence.
    if "." in stripped and "." not in entity:
        stripped = stripped.split(".")[-1].strip()

    # Finally, if we really messed something up along the way, just default to the
    # cleaned up paraphrase prompt.
    if entity not in stripped:
        logger.debug(f"prompt cleaning failed for {entity}: {stripped}")
        return prompt

    return stripped
    
def _reformat_counterfact_sample(cf_sample: dict) -> ContextMediationSample:
    """Reformat the counterfact sample."""
    cf_case_id = cf_sample["case_id"]
    cf_requested_rewrite = cf_sample["requested_rewrite"]
    cf_subject = cf_requested_rewrite["subject"]
    cf_target_new = cf_requested_rewrite["target_new"]["str"]
    cf_target_true = cf_requested_rewrite["target_true"]["str"]
    cf_prompt = cf_requested_rewrite["prompt"].format(cf_subject)
    cf_paraphrase_prompts = cf_sample["paraphrase_prompts"]

    entity = cf_subject
    prompt = _strip_counterfact_paraphrase_prompt(entity, cf_paraphrase_prompts[0])
    context = f"{cf_prompt} {cf_target_new}"
    attribute = context.split(entity)[-1].strip(",-;: ")
    target_mediated = cf_target_new
    target_unmediated = cf_target_true

    return ContextMediationSample(
        id=str(cf_case_id),
        entity=entity,
        prompt=prompt,
        context=context,
        attribute=attribute,
        target_mediated=target_mediated,
        target_unmediated=target_unmediated,
        # NOTE(evandez): Need to copy or else remove_columns will directly
        # delete keys on the original dict, causing source to be empty dict.
        source={**cf_sample},
    )
    
def _reformat_counterfact_file(intermediate_file: str, target_file: str):
    """Reformat the counterfact file to be jsonl instead of json."""
    with open(intermediate_file, "r") as f:
        lines = json.load(f)
    with open(target_file, "w") as f:
        for line in tqdm(lines, desc="reformat counterfact"):
            json.dump(_reformat_counterfact_sample(line), f)

def download_file(file: str, url: str):
    """Download the url to file."""
    logger.info(f"Downloading {file} from {url}...")
    file = Path(file)
    file.parent.mkdir(exist_ok=True, parents=True)
    wget.download(url, out=str(file))

def download_attribute_snippets(file_folder: str):    
    file = os.path.join(file_folder, "attribute_snippets.json")
    if not os.path.exists(file):
        download_file(file, ATTRIBUTE_SNIPPETS_URL)
        
def download_tfidf_vectorizer(file_folder: str):    
    tf_idf_vocab_file = os.path.join(file_folder, "tfidf_vocab.json")
    if not os.path.exists(tf_idf_vocab_file):
        download_file(tf_idf_vocab_file, TFIDF_VOCAB_URL)
    idf_file = os.path.join(file_folder, "idf.npy")
    if not os.path.exists(idf_file):
        download_file(idf_file, TFIDF_IDF_URL)
        
def download_data(file_folder: str):
    target_file = "counterfact.jsonl"
    target_file_path = os.path.join(file_folder, target_file)
    if not os.path.exists(target_file_path):
        intermediate_file = "counterfact.json"
        intermediate_file_path = os.path.join(file_folder, intermediate_file)
        if not os.path.exists(intermediate_file_path):
            download_file(intermediate_file_path, COUNTERFACT_URL)
        _reformat_counterfact_file(intermediate_file_path, target_file_path)

def main(args: argparse.Namespace):
    file_folder: str = args.path
    download_data(file_folder)
    download_attribute_snippets(file_folder)
    download_tfidf_vectorizer(file_folder)
    
if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to download")
    args = parser.parse_args()
    main(args)