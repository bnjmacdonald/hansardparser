
import os
import re
import json
import warnings
import pandas as pd
from google.cloud import storage

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import config
from .utils import extract_flatworld_tags


@registry.register_problem
class HansardLineHasSpeaker(text_problems.Text2ClassProblem):
    """Predicts whether a line from a Hansard transcript contains a speaker name."""

    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER


    @property
    def is_generate_per_split(self):
        return False


    @property
    def num_classes(self):
        """The number of classes."""
        return 2


    def class_labels(self, data_dir):
        """String representation of the classes."""
        del data_dir
        return ['no', 'yes']


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
        lines = pd.read_csv(dataset_path)
        assert lines.text.isnull().sum() == 0, "Every line should have a non-null text value."
        for _, line in lines.iterrows():
            line_text = line['text']
            if config.RM_FLATWORLD_TAGS:
                # removes Flatworld tags from lines.
                line_text, _ = extract_flatworld_tags(line_text)
            # label: 1 if 'start' or 'end' column is not null; 0 otherwise.
            label = int(pd.notnull(line['start']) or pd.notnull(line['end']))
            yield {'inputs': line_text, 'label': label}
