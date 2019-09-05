# coding: utf8
from __future__ import unicode_literals

import random
import prodigy
from prodigy.components.loaders import JSONL
from prodigy.models.ner import EntityRecognizer
from prodigy.components.preprocess import split_sentences
from prodigy.util import split_string

import spacy
from spacy.kb import KnowledgeBase

# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe('entity_linker.match_entities',
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    kb_path=("Path to the KB, its vocab corresponding to the spacy_model", "positional", None, str),
    label=("One or more comma-separated labels", "option", "l", split_string),
    exclude=("Names of datasets to exclude", "option", "e", split_string),
    unsegmented=("Don't split sentences", "flag", "U", bool)
)
def entity_linker_match_entities(dataset, spacy_model, source, kb_path, label=None,
                                 exclude=None, unsegmented=False):
    """
    Suggest entities predicted by the EntityRecognizer, and mark whether they
    are examples of the entity you're interested in.
    """
    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    # Load the spaCy model
    nlp = spacy.load(spacy_model)

    # Load the knowledge base
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=1)
    kb.load_bulk(str(kb_path))

    # Initialize Prodigy's entity recognizer model, which uses beam search to
    # find all possible analyses and outputs (score, example) tuples
    model = EntityRecognizer(nlp, label=label)

    if not unsegmented:
        # Use spaCy to split text into sentences
        stream = split_sentences(nlp, stream)

    # Apply the NER to the stream
    # Filter out the scores to only yield the examples for annotations.
    stream = (eg for score, eg in model(stream))

    stream = add_options(stream, kb)  # add options to each task

    return {
        'view_id': 'choice',    # Annotation interface to use
        'dataset': dataset,     # Name of dataset to save annotations
        'stream': stream,       # Incoming stream of examples
        'exclude': exclude,     # List of dataset names to exclude
        'config': {             # Additional config settings, mostly for app UI
            'lang': nlp.lang,
            "choice_auto_accept": True
        }
    }


def add_options(stream, kb):
    """Helper function to add options to every task in a stream. It takes the annotated
    span from the EntityRecognizer and adds suitable candidates from the KB as options."""

    for task in stream:
        # assume there is only one Span annotated, and use its text to generate NEL candidates
        mention = task["spans"][0]["text"]
        candidates = kb.get_candidates(mention)
        options = []
        # don't yield the task if there are no appropriate candidates from the KB
        if candidates:
            # randomly shuffle the candidates to avoid bias
            random.shuffle(candidates)
            for c in candidates:
                # TODO: get URL from KB instead of hardcoded here
                url = "<a href='https://www.wikidata.org/wiki/" + c.entity_ + "'>" + c.entity_ + "</a>"
                options.append({"id": c.entity_, "html": url})

            task["options"] = options
            yield task
