"""Reformat datasets.

This script handles reformatting and caching dataset from outside sources.
Callers usually should specify --dataset-file and point to the raw data
downloaded from its source.

Once the data is reformatted, you'll no longer have to specify --dataset-file
to any other scripts. The code will simply read it from the cache.
"""
import json
import argparse
from pathlib import Path
import pickle
from tqdm import tqdm

from benchmarks.utils.pasta_utils import ContextMediationSample, setup_logger

import spacy
nlp = spacy.load("en_core_web_sm")

import logging
logger = logging.getLogger(__name__)

_BIOS_BIAS_BLACKLISTED_NAMES = frozenset(
    {
        "Non-Residential",
    }
)

# These prefixes do not make as much sense when put in front of the first name, so
# we'll try to remove them as much as possible.
_BIOS_BIAS_PREFIXES = (
    "professor",
    "prof.",
    "prof",
    "dr.",
    "dr",
    "doctor",
    "mr.",
    "mr",
    "ms.",
    "ms",
    "mrs.",
    "mrs",
    "rev.",
    "rev",
    "pastor",
)

def _get_attribute(
    bb_bio:str, 
    bb_name:str,
    nlp, 
    sent_idx: int|None=None, 
):
    if sent_idx is None:
        attribute = bb_bio[bb_bio.index(bb_name) + len(bb_name) :].strip(".: ")
    else:
        doc = nlp(bb_bio)
        sents = [sent for sent in doc.sents]
        attr_sent =  str(sents[sent_idx])
        if sent_idx == 0 and bb_name in attr_sent:
            attribute = attr_sent[attr_sent.index(bb_name) + len(bb_name) :].strip(".: ")
        else:
            attribute = attr_sent
    return attribute

def _reformat_bias_in_bios_file(
    pkl_file: Path,
    bio_min_words: int = 10,
    sent_min_words: int = 3,
    limit: int | None = 50000,
    file_name: str = "biosbias.json",
    sents_choice: int | str = 1, 
    attr_sent_idx: int | None = None, 
) -> Path:
    """Reformat the Bias in Bios pickle file on disk."""
    with pkl_file.open("rb") as handle:
        data = pickle.load(handle)

    if limit is not None:
        data = data[:limit]

    # We'll use first name as the entity.
    bb_names = [sample["name"][0] for sample in data]

    # Take only one sentence of each bio to make the task harder.
    nlp = spacy.load("en_core_web_sm")
    bb_bios_raw = [sample["raw"] for sample in data]

    bb_bios_abridged = []
    for doc in tqdm(nlp.pipe(bb_bios_raw), total=len(data), desc="parse biosbias"):
        sents = [
            str(sent)
            for sent in doc.sents
            if len(str(sent).strip().split()) >= sent_min_words
        ]
        if len(sents) == 0:
            bb_bio_abridged = ""  # will get filtered out later
        elif len(sents) == 1:
            bb_bio_abridged = sents[0]  # no choice but first
        else:
            if sents_choice == "all":
                bb_bio_abridged = " ".join(sents)
            else:
                bb_bio_abridged = sents[sents_choice]  # otherwise, always take second sentence
        bb_bios_abridged.append(bb_bio_abridged)

    # Normalize the samples.
    lines = []
    for index, (sample, bb_name, bb_bio) in enumerate(
        tqdm(zip(data, bb_names, bb_bios_abridged), total=len(data), desc="process")
    ):
        if bb_name in _BIOS_BIAS_BLACKLISTED_NAMES:
            logger.debug(
                f"will not include sample #{index} because it has "
                f"blacklisted name '{bb_name}'"
            )
            continue

        # Replace all variants of the person's name with their first name.
        # We'll ignore suffixes, but try to strip prefixes since they are much more
        # common and do not mesh well with just first names.
        full_name = " ".join(part for part in sample["name"] if part)
        last_name = sample["name"][-1]
        first_last_name = f"{bb_name} {last_name}"

        replacements = []
        for name in (full_name, first_last_name, last_name):
            for prefix in _BIOS_BIAS_PREFIXES:
                replacements.append(f"{prefix} {name}")
                replacements.append(f"{prefix.capitalize()} {name}")
                replacements.append(f"{prefix.upper()} {name}")
        replacements += [full_name, first_last_name, last_name]
        for replace in replacements:
            bb_bio = bb_bio.replace(replace, bb_name)
        bb_bio = bb_bio.strip("*• ")

        bb_title = sample["title"].replace("_", " ")
        bb_id = "_".join(part for part in sample["name"] if part)

        # NOTE(evan): This filtering is necessary due to a downstream limitation where
        # the editor assumes that the last occurrence of an entity is always the second
        # occurrence.
        n_occurrences = bb_bio.count(bb_name)
        if n_occurrences == 0:
            bb_bio = f"About {bb_name}: {bb_bio}"
        elif n_occurrences != 1:
            logger.debug(
                f"will not include sample #{index} because there are "
                f"{n_occurrences} (!= 1) occurrences of '{bb_name}' in '{bb_bio}'"
            )
            continue

        approx_n_words = len(bb_bio.split())
        if approx_n_words < bio_min_words:
            logger.debug(
                f"will not include sample #{index} because bio '{bb_bio}' contains "
                f"too few words (approx. < {bio_min_words})"
            )
            continue

        entity = bb_name
        prompt = f"{entity} has the occupation of a/an "
        context = bb_bio
        # attribute = bb_bio[bb_bio.index(bb_name) + len(bb_name) :].strip(".: ")
        attribute = _get_attribute(bb_bio, bb_name, nlp, sent_idx=attr_sent_idx)
        target_mediated = bb_title

        line = ContextMediationSample(
            id=bb_id,
            source=sample,
            entity=entity,
            prompt=prompt,
            context=context,
            attribute=attribute,
            target_mediated=target_mediated,
            target_unmediated=None,
        )
        lines.append(line)

    # Save in jsonl format.
    json_file = Path(file_name)
    # create folder if it does not exist
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with json_file.open("w") as handle:
        for line in lines:
            json.dump(dict(line), handle)
    return json_file


def main(args: argparse.Namespace) -> None:
    """Do the reformatting by loading the dataset once."""
    _reformat_bias_in_bios_file(
        args.biasbios_raw_path, bio_min_words=10, sent_min_words=3, 
        file_name=args.biasbios_save_file, sents_choice="all", 
        attr_sent_idx=0,
    )

if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser(description="reformat datasets on disk")
    parser.add_argument(
        "--biasbios_raw_path", type=Path, help="path to BiasBios raw file.", default=None, 
    )
    parser.add_argument(
        "--biasbios_save_file", type=Path, help="path to save the dataset.", default=None, 
    )
    args = parser.parse_args()
    main(args)
