import os
import unittest
import logging
from copy import copy
from pathlib import Path
import codecs
import json

# required for tensorflow logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from finetune import SequenceLabeler
from finetune.utils import indico_to_finetune_sequence, finetune_to_indico_sequence
from finetune.metrics import (
    sequence_labeling_token_precision, sequence_labeling_token_recall,
    sequence_labeling_overlap_precision, sequence_labeling_overlap_recall
)
import requests
from bs4 import BeautifulSoup as bs
from bs4.element import Tag

def _download_reuters():
    """
    Download Stanford Sentiment Treebank to enso `data` directory
    """
    path = Path(dataset_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(dataset_path):
        url = "https://raw.githubusercontent.com/dice-group/n3-collection/master/reuters.xml"
        r = requests.get(url)
        with open(dataset_path, "wb") as fp:
            fp.write(r.content)

    with codecs.open(dataset_path, "r", "utf-8") as infile:
        soup = bs(infile, "html5lib")

    docs = []
    docs_labels = []
    for elem in soup.find_all("document"):
        texts = []
        labels = []

        # Loop through each child of the element under "textwithnamedentities"
        for c in elem.find("textwithnamedentities").children:
            if type(c) == Tag:
                if c.name == "namedentityintext":
                    label = "Named Entity"  # part of a named entity
                else:
                    label = "<PAD>"  # irrelevant word
                texts.append(c.text)
                labels.append(label)

        docs.append(texts)
        docs_labels.append(labels)

    with open(processed_path, 'wt') as fp:
        json.dump((docs, docs_labels), fp)

n_sample = 100
n_hidden = 768
dataset_path = os.path.join(
    '/media/liah/DATA/reuters', 'reuters.xml'
)
processed_path = os.path.join('/media/liah/DATA/reuters', 'reuters.json')

_download_reuters()

with open(processed_path, 'rt') as fp:
    texts, labels = json.load(fp)

tf.reset_default_graph()

model = SequenceLabeler(batch_size=2, max_length=256, lm_loss_coef=0.0, verbose=False)

raw_docs = ["".join(text) for text in texts]
# raw_docs : list of article str
# texts : is raw_docs
# annotations : list of list of dict, each dict is a {start, end, label, text} of an NE, each article contains several
#               such dict

texts, annotations = finetune_to_indico_sequence(raw_docs, texts, labels)
train_texts, test_texts, train_annotations, test_annotations = train_test_split(texts, annotations, test_size=0.1)
# model.fit(train_texts)
model.fit(train_texts, train_annotations)
predictions = model.predict(test_texts)
probas = model.predict_proba(test_texts)
print(probas)