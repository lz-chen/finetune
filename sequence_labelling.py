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
    sequence_labeling_overlap_precision, sequence_labeling_overlap_recall,
    sequence_labeling_micro_token_f1, sequence_labeling_overlaps
)
from pprint import pprint
import requests
from bs4 import BeautifulSoup as bs
from bs4.element import Tag
from random import shuffle


def get_text_and_annotations_from_file(dataset_folder: Path):
    ner_texts = []
    ner_annotations = []
    unsorted_annotation_files = dataset_folder.glob('*.annotation.json')
    unsorted_text_files = dataset_folder.glob('*.text.json')
    annotation_files = sorted([x for x in unsorted_annotation_files])
    text_files = sorted([x for x in unsorted_text_files])
    for text_file, annotation_file in zip(text_files, annotation_files):
        assert text_file.name.split('.')[0] == annotation_file.name.split('.')[0]
        textf = dataset_folder.joinpath(text_file)
        annotationf = dataset_folder.joinpath(annotation_file)
        with textf.open('r') as tf, annotationf.open('r') as af:
            for i, (line1, line2) in enumerate(zip(tf, af)):
                # if i > 100:
                #     break
                text = json.loads(line1)
                annotations = json.loads(line2)
                if annotations != [] and text != '':
                    ner_texts.append(text)
                    ner_annotations.append(annotations)

    idxs = [x for x in range(len(ner_texts))]
    shuffle(idxs)
    ner_texts = [ner_texts[i] for i in idxs]
    ner_annotations = [ner_annotations[i] for i in idxs]
    return ner_texts, ner_annotations


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

# _download_reuters()

# with open(processed_path, 'rt') as fp:
#     texts, labels = json.load(fp)

tf.reset_default_graph()

model = SequenceLabeler(batch_size=32, max_length=256, lm_loss_coef=0.0, verbose=False)

# raw_docs = ["".join(text) for text in texts]
# raw_docs : list of article str
# texts : is raw_docs
# annotations : list of list of dict, each dict is a {char_start, char_end, label, text} of an NE, each article
#               contains several such dict
# texts, annotations = finetune_to_indico_sequence(raw_docs, texts, labels)

dataset_folder: Path = Path('/media/liah/DATA/acme_data_ner/dataset_ner_en')
texts, annotations = get_text_and_annotations_from_file(dataset_folder=dataset_folder)
data_size = len(texts)
train_texts, test_texts, train_annotations, test_annotations = train_test_split(texts[:data_size],
                                                                                annotations[:data_size],
                                                                                test_size=0.1)

model_store_path = '/media/liah/DATA/acme_data_ner/models/model_{:d}'.format(data_size)
model_log_path = '/media/liah/DATA/acme_data_ner/log/model_{:d}'.format(data_size)

# model.fit(train_texts) # unsup. learning for lm
model.fit(train_texts, train_annotations)
model.save(model_store_path)

predictions = model.predict(test_texts)
probas = model.predict_proba(test_texts)
overlaps = sequence_labeling_overlaps(test_annotations, predictions)
f1_score = sequence_labeling_micro_token_f1(test_annotations, predictions)

with open(model_log_path, 'a') as logf:
    pprint(probas, logf)
    pprint(f1_score, logf)
    pprint('TP, FP, TN, FN:', logf)
    pprint(overlaps, logf)

