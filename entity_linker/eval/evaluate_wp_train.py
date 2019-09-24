# coding: utf8
from __future__ import unicode_literals

import json
from pathlib import Path

import spacy
from prodigy.util import write_jsonl

""" Code to turn the WP training dataset into evaluation tasks for error analysis.
Basically this is to assess how correct the automatically generated data is. """

kb_dir = "C:/Users/Sofie/Documents/data/spacy_test_CLI_KB/"
training_dir = Path("C:/Users/Sofie/Documents/data/spacy_test_CLI_train_dataset/")
output_file = Path("./data/dev_training.jsonl")

annotations_file = Path("./data/eval_wp_el_output_310.jsonl")


def is_dev(article_id):
    return article_id.endswith("3")


def write(limit=None):
    nlp = spacy.load(Path(kb_dir) / "nlp")

    cnt_articles = 0
    cnt_entities = 0
    text_dicts = []
    for textfile in training_dir.iterdir():
        if textfile.name.endswith(".txt"):
            article_id = textfile.name.split(".")[0]
            if is_dev(article_id):
                if not limit or cnt_articles < limit:
                    cnt_articles += 1
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
    print("Analysed", cnt_articles, "dev articles")
    print("Found", cnt_entities, "entities")


def read():
    with output_file.open("r", encoding="utf8") as json_file:
        json_list = list(json_file)
        print("Read", len(json_list), "lines")

    articles = set()
    mentions = 0

    for json_str in json_list:
        # print(json_str)
        result = json.loads(json_str)
        article_id = result["article_id"]
        articles.add(article_id)
        text = result["text"]
        for span in result["spans"]:
            parsed_wp = span["parsed_WP_ID"]
            start_char = int(span["start"])
            end_char = int(span["end"])
            mention = text[start_char:end_char]
            # print("found mention", mention, "parsed", parsed_wp)
            mentions += 1

    print("Total:", mentions, "annotations in", len(articles), "articles")


def analyse():
    print("Running evaluations for", annotations_file)
    with annotations_file.open("r", encoding="utf8") as json_file:
        json_list = list(json_file)
        print("read", len(json_list), "lines")

    nil_counts = {}
    sameQ = 0
    differentQ = 0
    ignore = 0
    total = 0
    articles = set()

    print()
    for json_str in json_list:
        result = json.loads(json_str)
        article_id = result["article_id"]
        articles.add(article_id)
        total += 1

        # assume there is only one span per line
        assert len(result["spans"]) == 1
        span = result["spans"][0]
        wp_id = span["parsed_WP_ID"]

        # ignoring "ignore" answers with zero length "accept" annotations
        if len(result["accept"]) == 0:
            ignore += 1
        else:
            assert len(result["accept"]) == 1
            sofie_id = result["accept"][0]
            answer = result["answer"]
            # print(article_id, "WP parsed", wp_id, "- Sofie annotated", sofie_id, "with", answer)
            if answer == "accept":
                if wp_id.startswith("Q") and sofie_id.startswith("Q"):
                    if wp_id == sofie_id:
                        sameQ += 1
                    else:
                        text = result["text"]
                        start_char = int(span["start"])
                        end_char = int(span["end"])
                        mention = text[start_char:end_char]

                        print("Manually annotated", sofie_id, "but got", wp_id,
                              "for mention", mention, "in article", article_id, "with text:", text)
                        differentQ += 1
                else:
                    nil_counts[sofie_id] = nil_counts.get(sofie_id, 0) + 1
            else:
                print("accept was", answer)

    print()
    print("Total:", total, "annotations in", len(articles), "articles")
    print("Found same:", sameQ)
    print("Found different:", differentQ)
    print("Found ignored:", ignore)
    print("Found NIL:", sum([y for x,y in nil_counts.items()]))
    for x, y in nil_counts.items():
        print(" - " + x + ":", y)


if __name__ == "__main__":
    # STEP 0: first write the JSONL from the original training files
    # write(limit=1000)

    # STEP 1: read the created JSONL to be sure it's OK
    # read()

    # STEP 2: run the actual annotations with the Prodigy recipe "entity_linker.eval"

    # STEP 3: now run the stats on the manual annotations
    analyse()
