# coding: utf8
from __future__ import unicode_literals

import random
import csv
from pathlib import Path

import prodigy
import spacy
from prodigy.components.loaders import JSONL
from prodigy.models.ner import EntityRecognizer
from prodigy.util import split_string, set_hashes

# TODO: get URL from KB instead of hardcoded here
URL_PREFIX = "https://www.wikidata.org/wiki/"

# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
from spacy.kb import KnowledgeBase


@prodigy.recipe(
    "entity_linker.annotate",
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    kb_dir=("Path to the KB dir", "positional", None, Path),
    label=("One or more comma-separated labels", "option", "l", split_string),
    exclude=("Names of datasets to exclude", "option", "e", split_string),
)
def entity_linker_eval(dataset, source, kb_dir, label=None, exclude=None):
    """
    Load a dataset of sentences, add NER + candidates from the KB,
    and offer each annotation as an evaluation task.
    """
    # Load the knowledge base
    nlp_dir = kb_dir / "nlp"

    nlp = spacy.load(nlp_dir)
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=1)
    kb.load_bulk(str(kb_dir / "kb"))

    # Initialize Prodigy's entity recognizer model, which uses beam search to
    # find all possible analyses and outputs (score, example) tuples
    model = EntityRecognizer(nlp, label=label)

    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    # we don't need to split sentences anymore, but we do need hashes
    stream = [set_hashes(eg) for eg in stream]

    # Apply the NER to the stream
    # Filter out the scores to only yield the examples for annotations.
    stream = (eg for score, eg in model(stream))

    # Read entity descriptions for printing a clear string
    loc_entity_desc = kb_dir / "entity_descriptions.csv"
    id_to_desc = dict()
    with loc_entity_desc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter="|")
        # skip header
        next(csvreader)
        for row in csvreader:
            id_to_desc[row[0]] = row[1]

    # add KB options to each task
    stream = add_options(stream, kb, id_to_desc)

    return {
        "view_id": "choice",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "exclude": exclude,  # List of dataset names to exclude
        "config": {"choice_auto_accept": True},  # Additional config settings,
    }


def add_options(stream, kb, id_to_desc):
    """Helper function to add options to every task in a stream. It takes the annotated
    span from the EntityRecognizer and adds suitable candidates from the KB as options."""

    for task in stream:
        text = task["text"]
        for span in task["spans"]:
            start_char = int(span["start"])
            end_char = int(span["end"])
            mention = text[start_char:end_char]

            # add candidates from the KB and include generic answers
            candidates = kb.get_candidates(mention)

            options = []
            for c in candidates:
                url = _print_url_option(c.entity_, id_to_desc)
                options.append({"id": c.entity_, "html": url})

            # randomly shuffle the candidates to avoid bias
            random.shuffle(options)

            options.append({"id": "NIL_otherLink", "text": "Link not in options"})
            options.append({"id": "NIL_ambiguous", "text": "Need more context"})
            options.append({"id": "NIL_noNE", "text": "Not a named entity"})
            options.append({"id": "NIL_noSentence", "text": "Not a proper sentence"})
            options.append({"id": "NIL_unsure", "text": "Unsure"})

            task["options"] = options
            yield task


def _print_url_option(entity_id, id_to_desc):
    descr = entity_id + ": " + id_to_desc.get(entity_id, "No description")
    url = "<a href='" + URL_PREFIX + entity_id + "'>" + descr + "</a>"
    return url
