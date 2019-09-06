# coding: utf8
from __future__ import unicode_literals

import random
import csv
from pathlib import Path

import prodigy
import spacy
from prodigy.components.loaders import JSONL
from prodigy.components.db import connect
from prodigy.util import split_string

from spacy.util import itershuffle

# TODO: get URL from KB instead of hardcoded here
URL_PREFIX = "https://www.wikidata.org/wiki/"

# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
from spacy.kb import KnowledgeBase


@prodigy.recipe(
    "entity_linker.eval",
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    kb_dir=("Path to the KB dir", "positional", None, Path),
    exclude=("Names of datasets to exclude", "option", "e", split_string),
    resume=("Resume from existing dataset", "flag", "R", bool),
)
def entity_linker_eval(dataset, source, kb_dir, exclude=None, resume=False):
    """
    Load a dataset of EL annotations, add additional candidates from the KB,
    and offer each annotation as an evaluation task.
    """
    # Load the knowledge base
    nlp_dir = kb_dir / "nlp"

    nlp = spacy.load(nlp_dir)
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=1)
    kb.load_bulk(str(kb_dir / "kb"))

    # Read entity descriptions for printing a clear string
    loc_entity_desc = kb_dir / "entity_descriptions.csv"
    id_to_desc = dict()
    with loc_entity_desc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter="|")
        # skip header
        next(csvreader)
        for row in csvreader:
            id_to_desc[row[0]] = row[1]

    if resume:
        # Connect to the database using the settings from prodigy.json
        DB = connect()
        if dataset and dataset in DB:
            # Get the existing annotations
            existing = DB.get_dataset(dataset)

    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    # add KB options to each task
    stream = add_options(stream, kb, id_to_desc)

    # shuffle the stream to mix up the annotations & articles
    shuffled_stream = itershuffle(stream, bufsize=1000)

    return {
        "view_id": "choice",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": shuffled_stream,  # Incoming stream of examples
        "exclude": exclude,  # List of dataset names to exclude
        "config": {"choice_auto_accept": True},  # Additional config settings,
    }


def add_options(stream, kb, id_to_desc):
    """Helper function to add options to every task in a stream. It takes the annotated
    span from the EntityRecognizer and adds suitable candidates from the KB as options."""

    for task in stream:
        text = task["text"]
        for span in task["spans"]:
            parsed_wp = span["parsed_WP_ID"]
            start_char = int(span["start"])
            end_char = int(span["end"])
            mention = text[start_char:end_char]

            # add candidates from the KB and include generic answers
            candidates = kb.get_candidates(mention)
            found_parsed = False

            options = []
            for c in candidates:
                if c.entity_ == parsed_wp:
                    found_parsed = True
                url = _print_url_option(c.entity_, id_to_desc)
                options.append({"id": c.entity_, "html": url})

            if not found_parsed:
                url = _print_url_option(parsed_wp, id_to_desc)
                options.append({"id": parsed_wp, "html": url})

            # randomly shuffle the candidates to avoid bias
            random.shuffle(options)

            options.append({"id": "NIL_otherLink", "html": "Link not in options"})
            options.append({"id": "NIL_ambiguous", "html": "Need more context"})
            options.append({"id": "NIL_noNE", "html": "Not a named entity"})
            options.append({"id": "NIL_noSentence", "html": "Not a proper sentence"})
            options.append({"id": "NIL_unsure", "html": "Unsure"})

            task["options"] = options
            yield task


def _print_url_option(entity_id, id_to_desc):
    descr = entity_id + ": " + id_to_desc.get(entity_id, "No description")
    url = "<a href='" + URL_PREFIX + entity_id + "'>" + descr + "</a>"
    return url
