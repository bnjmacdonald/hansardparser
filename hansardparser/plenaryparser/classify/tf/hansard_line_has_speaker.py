
import os
import re
import json
import warnings
import pandas as pd

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

# from hansardparser import settings

VERBOSITY = 1
# TODO: get rid of this hard-coded file path.
DATA_ROOT = '/Users/bnjmacdonald/Documents/current/projects/hansardparser/data'
DATA_PATH = os.path.join(DATA_ROOT, 'tests', 'manual', 'speaker_name_hand_labels_w_text20.csv')
RM_FLATWORLD_TAGS = True


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
        if is_train:
            dataset_path = DATA_PATH
        else:
            dataset_path = DATA_PATH
        lines = pd.read_csv(dataset_path)
        assert lines.text.isnull().sum() == 0, "Every line should have a non-null text value."
        for _, line in lines.iterrows():
            line_text = line['text']
            if RM_FLATWORLD_TAGS:
                # removes Flatworld tags from lines.
                inner_regex = r'[/ ]{0,2}(header|newspeech|speech|sub-?header|scene|district)[/ ]{0,2}'
                line_text = re.sub(rf'(<{inner_regex}>)|(<{inner_regex})|({inner_regex}>)', '', line_text,
                    flags=re.IGNORECASE)
                if VERBOSITY > 1 and re.search(r'(<[/ \w]{3,})|([/ \w]{3,}>)', line_text):
                    warnings.warn(f'angle bracket exists in line: {line_text}')
            # label: 1 if 'start' or 'end' column is not null; 0 otherwise.
            label = int(pd.notnull(line['start']) or pd.notnull(line['end']))
            yield {'inputs': line_text, 'label': label}
