# Lint as: python3
"""REBEL-Persian"""

from __future__ import absolute_import, division, print_function
import time

import pandas as pd

import datasets

import re 
import json
import logging
import math
import os
from collections import defaultdict

_DESCRIPTION = """\
REBEL-Persian is a silver dataset created using the paper REBEL (Relation Extraction By End-to-end Language generation) for the Perisan language
"""

_URL = ""
_URLS = {
    "train": _URL + "train.jsonl",
    "dev": _URL + "val.jsonl",
    "test": _URL + "test.jsonl",
}


class PRebelConfig(datasets.BuilderConfig):
    """BuilderConfig for REBEL-Persian."""

    def __init__(self, **kwargs):
        """BuilderConfig for REBEL-Persian.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PRebelConfig, self).__init__(**kwargs)


class Prebel(datasets.GeneratorBasedBuilder):
    """Prebel 1.0"""

    BUILDER_CONFIGS = [
        PRebelConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "triplets": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            # homepage="",
#             citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.data_files:
            downloaded_files = {
                "train": self.config.data_files["train"], # self.config.data_dir + "en_train.jsonl",
                "dev": self.config.data_files["dev"], #self.config.data_dir + "en_val.jsonl",
                "test": self.config.data_files["test"], #self.config.data_dir + "en_test.jsonl",
            }
        else:
            downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        if isinstance(filepath, list):
            filepath = filepath[0]
        logging.info("generating examples from = %s", filepath)
        filedir = os.path.dirname(filepath)
        relations_filepath = os.path.join(filedir, 'relations_count.tsv')
        relations_df = pd.read_csv(relations_filepath, header = None, sep='\t')
        relations = list(relations_df[0])

        with open(filepath, encoding="utf-8") as f:
            uri_set = set()
            for id_, row in enumerate(f):
                article = json.loads(row)
                prev_len = 0
                if len(article['triples']) == 0:
                    continue
                if article['uri'] in uri_set:
                    #   (f"duplicate {article['title']}")
                    # time.sleep
                    continue
                uri_set.add(article['uri'])
                count = 0
                for text_paragraph in article['text'].split('\n'):
                    if len(text_paragraph) == 0:
                        continue
                    sentences = re.split(r'(?<=[.])\s', text_paragraph)
                    text = ''
                    for sentence in sentences:
                        text += sentence + ' '
                        if any([entity['boundaries'][0] < len(text) + prev_len < entity['boundaries'][1] for entity in article['entities']]):
                            continue
                        entities = sorted([entity for entity in article['entities'] if prev_len < entity['boundaries'][1] <= len(text)+prev_len], key=lambda tup: tup['boundaries'][0])
                        decoder_output = '<triplet> '
                        for int_ent, entity in enumerate(entities):
                            triplets = sorted([triplet for triplet in article['triples'] if triplet['subject'] == entity and prev_len< triplet['subject']['boundaries'][1]<=len(text) + prev_len and prev_len< triplet['object']['boundaries'][1]<=len(text)+ prev_len and triplet['predicate']['surfaceform'] in relations], key=lambda tup: tup['object']['boundaries'][0])
                            if len(triplets) == 0:
                                continue
                            decoder_output += entity['surfaceform'] + ' <subj> '
                            for triplet in triplets:
                                decoder_output += triplet['object']['surfaceform'] + ' <obj> '  + triplet['predicate']['surfaceform'] + ' <subj> '
                            decoder_output = decoder_output[:-len(' <subj> ')]
                            decoder_output += ' <triplet> '
                        decoder_output = decoder_output[:-len(' <triplet> ')]
                        count += 1
                        prev_len += len(text)

                        if len(decoder_output) == 0:
                            text = ''
                            continue

                        text = re.sub('([\[\].,!?()])', r' \1 ', text.replace('()', ''))
                        text = re.sub('\s{2,}', ' ', text)

                        yield article['uri'] + '-' + str(count), {
                            "title": article['title'],
                            "context": text,
                            "id": article['uri'] + '-' + str(count),
                            "triplets": decoder_output,
                        }
                        text = ''
