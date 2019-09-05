# coding: utf8
from __future__ import unicode_literals

import json
from pathlib import Path

import spacy
from prodigy.util import write_jsonl

""" Code to turn the WP training dataset into evaluation tasks for error analysis.
Basically this is to assess how correct the automatically generated data is. """

kb_dir = "C:/Users/Sofie/Documents/data/spacy_test_CLI_KB/"
training_dir = Path("C:/Users/Sofie/Documents/data/spacy_eval_train/")
output_file = Path("./training_100.jsonl")


def write():
    nlp = spacy.load(Path(kb_dir) / "nlp")

    cnt_articles = 0
    cnt_entities = 0
    text_dicts = []
    for textfile in training_dir.iterdir():
        if textfile.name.endswith(".txt"):
            cnt_articles += 1
            article_id = textfile.name.split(".")[0]
            text = None
            with textfile.open("r", encoding="utf8") as f:
                try:
                    text = f.read()
                except Exception as e:
                    print("Problem parsing article", article_id, e)

            doc = nlp(text)

            gold_file_loc = article_id + "_gold.csv"
            entityfile_loc = training_dir / gold_file_loc
            with entityfile_loc.open("r", encoding="utf8") as entityfile:
                for line in entityfile:
                    fields = line.replace("\n", "").split(sep="|")
                    article_id = fields[0]
                    alias = fields[1]
                    wd_id = fields[2]
                    if article_id != "article_id":
                        entity_start = int(fields[3])
                        entity_end = int(fields[4])

                        # using sentence boundaries (or default to adding 200 characters)
                        text_start = max(0, entity_start-100)
                        text_end = min(len(text), entity_end+100)

                        for sent in doc.sents:
                            if sent.start_char <= entity_start and sent.end_char >= entity_end:
                                text_start = sent.start_char
                                text_end = sent.end_char

                        # change the offsets to the sentence
                        span_dict = {
                            "start": entity_start - text_start,
                            "end": entity_end - text_start,
                            "parsed_WP_ID": wd_id,
                        }

                        text_dicts.append({"article_id": article_id,
                                           "text": text[text_start:text_end],
                                           "spans": [span_dict],
                                           "sent_offset": text_start})
                        cnt_entities += 1
    write_jsonl(output_file, text_dicts)
    print("Analysed", cnt_articles, "articles")
    print("Found", cnt_entities, "entities")


def read():
    with output_file.open("r", encoding="utf8") as json_file:
        json_list = list(json_file)
        print("read", len(json_list))

    for json_str in json_list:
        print(json_str)
        result = json.loads(json_str)
        text = result["text"]
        for span in result["spans"]:
            parsed_wp = span["parsed_WP_ID"]
            start_char = int(span["start"])
            end_char = int(span["end"])
            mention = text[start_char:end_char]
            print("found mention", mention, "parsed", parsed_wp)


if __name__ == "__main__":
    write()
    # read()
