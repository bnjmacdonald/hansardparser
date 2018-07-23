
import os
import re
import json
import warnings
import pandas as pd

import tensorflow as tf
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
from . import config
from .utils import extract_flatworld_tags

# upsample rare classes.
UPSAMPLE = True
# random seed for upsampling rare classes.
SEED = 933707

@registry.register_problem
class HansardLineType4(text_problems.Text2ClassProblem):
    f"""Predicts the type of each line in a Hansard transcript.
    
    Line types: {list(config.LINE_LABEL_CODES.keys())}.

    If you want to serve a trained model that accepts strings as inputs, use
    `HansardLineType4Predict`.

    If you want to serialize unlabeled data, use `HansardLineType4Predict`.
    """

    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

    @property
    def is_generate_per_split(self):
        return True
    
    @property
    def num_classes(self):
        """The number of classes."""
        return len(config.LINE_LABEL_CODES)
    
    def class_labels(self, data_dir):
        """String representation of the classes."""
        del data_dir
        return sorted(config.LINE_LABEL_CODES.keys(), key=lambda x: config.LINE_LABEL_CODES[x])

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        is_train = dataset_split == problem.DatasetSplit.TRAIN
        assert 'train.csv' in os.listdir(tmp_dir)
        if is_train:
            dataset_path = os.path.join(tmp_dir, 'train.csv')
        else:
            dataset_path = os.path.join(tmp_dir, 'dev.csv')
        samples = self._generate_csv_samples(dataset_path, is_train)
        for sample in samples:
            yield sample
    

    def _generate_csv_samples(self, dataset_path, is_train):
        lines = pd.read_csv(dataset_path)
        # TODO: skip lines with a note.
        # keeps only lines where label exists and is not equal to 'blank'.
        lines = lines[(lines.label.notnull()) & (~lines.label.isin(['blank']))]
        assert lines.text.isnull().sum() == 0, "Every line should have a non-null text value."
        if UPSAMPLE and is_train:
            lines['label_prop'] = lines.groupby('label').label.transform(lambda x: x.shape[0] / float(lines.shape[0]))
            lines['weight'] = lines.shape[0] / lines['label_prop']
            lines['weight'] /= lines['weight'].sum()
            assert lines['weight'].isnull().sum() == 0
            # sanity checks:
            # pd.crosstab(lines['label'], lines['label_prop'])
            # lines['label'].value_counts() / lines.shape[0]
            # pd.crosstab(lines['label'], lines['weight'])
            # samples 10x the number of examples in `lines`.
            lines = lines.sample(n=lines.shape[0] * 10, replace=True, weights=lines.weight.values, random_state=SEED)
            # sanity check: number of examples per label class should be roughly equal.
            assert all(abs(lines['label'].value_counts() / lines.shape[0] - 1.0 / lines['label'].nunique()) < 0.01), \
                ('There should be roughly equal numbers of examples in each label class after upsampling.')
        # lines = [{'text': l['text'], '_id': f"{l['_id']}-{l['line']}"} for ix, l in lines.iterrows()]
        for _, line in lines.iterrows():
            line_text = line['text']
            if config.RM_FLATWORLD_TAGS:
                # removes Flatworld tags from lines.
                line_text, _ = extract_flatworld_tags(line_text)
            yield {'inputs': line_text, 'label': config.LINE_LABEL_CODES[line['label']]}

