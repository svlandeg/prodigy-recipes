import spacy
from spacy.kb import KnowledgeBase

import prodigy
from prodigy.util import set_hashes
from prodigy.models.ner import EntityRecognizer
from prodigy.components.loaders import TXT
from prodigy.components.filters import filter_duplicates

import csv
from pathlib import Path


@prodigy.recipe(
    "entity_linker.manual",
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a .txt file", "positional", None, Path),
    nlp_dir=("Path to the NLP model with a pretrained NER component", "positional", None, Path),
    kb_loc=("Path to the KB", "positional", None, Path),
    entity_loc=("Path to the file with additional information about the entities", "positional", None, Path),
)
def entity_linker_manual(dataset, source, nlp_dir, kb_loc, entity_loc):
    # Set up the NLP model and the Knowledge base
    nlp = spacy.load(nlp_dir)
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=1)
    kb.load_bulk(kb_loc)
    model = EntityRecognizer(nlp)

    # Read the input sentences and apply the NER to it
    stream = TXT(source)
    stream = [set_hashes(eg) for eg in stream]
    stream = (eg for score, eg in model(stream))

    # Read entity descriptions for printing a bit more information than just the QID
    id_to_desc = dict()
    with entity_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            id_to_desc[row[0]] = row[2]

    stream = _add_options(stream, kb, id_to_desc)
    stream = filter_duplicates(stream, by_input=True, by_task=False)

    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "choice",
        "config": {"choice_auto_accept": True},
    }


def _add_options(stream, kb, id_to_desc):
    # Add the KB options to each task
    for task in stream:
        text = task["text"]
        for span in task["spans"]:
            start_char = int(span["start"])
            end_char = int(span["end"])
            mention = text[start_char:end_char]

            # add candidates from the KB and include generic answers
            candidates = kb.get_candidates(mention)
            if candidates:
                options = []
                for c in candidates:
                    url = _print_url_option(c.entity_, id_to_desc)
                    options.append({"id": c.entity_, "html": url})

                # sort the list by ID to ensure quicker annotation
                options = sorted(options, key=lambda r: int(r["id"][1:]))
                options.append({"id": "NIL_otherLink", "text": "Link not in options"})
                options.append({"id": "NIL_ambiguous", "text": "Need more context"})

                task["options"] = options
                yield task


def _print_url_option(entity_id, id_to_desc):
    url_prefix = "https://www.wikidata.org/wiki/"
    descr =  id_to_desc.get(entity_id, "No description")
    option = "<a href='" + url_prefix + entity_id + "'>" + entity_id + "</a>: " + descr
    return option
