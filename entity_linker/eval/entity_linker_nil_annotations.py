# coding: utf8
from __future__ import unicode_literals

import prodigy
from prodigy.components.loaders import JSONL
from prodigy.util import split_string, set_hashes


NER_LABELS_TO_IGNORE = ['CARDINAL', 'DATE', 'MONEY', 'ORDINAL', 'QUANTITY', 'TIME', 'PERCENT']

# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.


@prodigy.recipe(
    "entity_linker.annotate_nil",
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    exclude=("Names of datasets to exclude", "option", "e", split_string),
)
def entity_linker_eval(dataset, source, exclude=None):
    """
    Load a dataset of sentences, add free-text input field,
    and offer each annotation as an evaluation task.
    """

    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    # we don't need to split sentences anymore, but we do need hashes
    stream = [set_hashes(eg) for eg in stream]

    my_html = (
        "{{before_text}}<b><font color='purple'>{{entity}}</font></b>{{after_text}}<br /><br />"
        '<input type="text" class="input" placeholder="Type a custom answer..." />'
        '<button onClick="updateFromInput()">Update</button><br />Answer: {{user_text}}'
    )

    my_script = (
        "function updateFromInput() {"
        "const text = document.querySelector('.input').value; "
        "window.prodigy.update({ user_text: text });}"
    )

    return {
        "view_id": "html",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "exclude": exclude,  # List of dataset names to exclude
        "config": {
            "html_template": my_html,
            "javascript": my_script,
        },
    }
