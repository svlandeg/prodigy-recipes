# coding: utf8
from __future__ import unicode_literals

import json
import random
import pandas as pd
from pathlib import Path

import spacy
from prodigy.util import write_jsonl

news_dir = Path("C:/Users/Sofie/Documents/data/kaggle_all_the_news/articles1.csv")
output_file = Path("./data/news_annotations.jsonl")
annotations_file = Path("./data/annotate_news_output_200.jsonl")


def write(limit=None):
    cnt_sentences = 0
    text_dicts = []

    nlp = spacy.load("en_core_web_lg")

    # read all 50K articles and select one sentence at random from each
    df = pd.read_csv(news_dir, encoding="utf-8")
    for index, row in df.iterrows():
        if not limit or cnt_sentences < limit:
            text = row["content"]
            doc = nlp(text)
            sentences = [sent for sent in doc.sents if len(sent.text) > 10]
            random.shuffle(sentences)
            sent = sentences[0]

            text_dicts.append(
                {
                    "article_id": row["id"],
                    "text": sent.text,
                    "article_text": text,
                    "sent_offset": sent.start_char,
                }
            )
            cnt_sentences += 1

    print("dict", text_dicts)
    write_jsonl(output_file, text_dicts)

    print("Found", cnt_sentences, "sentences")
    print("Wrote to", output_file)


def analyse():
    # TODO: analyse by label
    print("Running evaluations for", annotations_file)
    with annotations_file.open("r", encoding="utf8") as json_file:
        json_list = list(json_file)
        print("read", len(json_list), "lines")

    nil_counts = {}
    qid = 0
    ignore = 0
    total = 0
    articles = set()

    print()
    for json_str in json_list:
        result = json.loads(json_str)
        article_id = result["article_id"]
        sent_text = result["text"]
        sent_offset = result["sent_offset"]
        art_text = result["article_text"]
        articles.add(article_id)
        total += 1

        # assume there is only one span per line
        assert len(result["spans"]) == 1
        span = result["spans"][0]
        span_start = span["start"]
        span_end = span["end"]
        span_text = span["text"]
        # print("Mention", span_text, "==",
        #      sent_text[span_start:span_end], "==",
        #      art_text[sent_offset+span_start:sent_offset+span_end])

        # ignoring "ignore" answers with zero length "accept" annotations
        if len(result["accept"]) == 0:
            ignore += 1
        else:
            assert len(result["accept"]) == 1
            sofie_id = result["accept"][0]
            answer = result["answer"]
            # print(article_id, "WP parsed", wp_id, "- Sofie annotated", sofie_id, "with", answer)
            if answer == "accept":
                if sofie_id.startswith("Q"):
                    qid += 1
                else:
                    nil_counts[sofie_id] = nil_counts.get(sofie_id, 0) + 1
            else:
                print("accept was", answer)

    print()
    print("Total:", total, "annotations in", len(articles), "articles")
    print("Found Q IDs:", qid)
    print("Found ignored:", ignore)
    print("Found NIL:", sum([y for x, y in nil_counts.items()]))
    for x, y in nil_counts.items():
        print(" - " + x + ":", y)


if __name__ == "__main__":
    # STEP 1: write the JSONL from the news snippets
    # write(limit=1000)

    # STEP 2: run the actual annotations with the Prodigy recipe "entity_linker.annotate"

    # STEP 3: now run the stats on the manual annotations
    analyse()
