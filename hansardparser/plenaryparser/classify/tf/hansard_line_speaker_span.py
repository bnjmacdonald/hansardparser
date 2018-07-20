
import os
import re
import json
import warnings
import pandas as pd

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems, text_encoder, generator_utils
from tensor2tensor.utils import registry

# from hansardparser import settings

# TODO: get rid of this hard-coded file path.
DATA_ROOT = '/Users/bnjmacdonald/Documents/current/projects/hansardparser/data'
DATA_PATH = os.path.join(DATA_ROOT, 'tests', 'manual', 'speaker_name_hand_labels_w_text20.csv')

CLASS_LABELS = ['O', 'B-speaker', 'I-speaker']


@registry.register_problem
class HansardLineSpeakerSpan(text_problems.Text2TextProblem):
    """Predict the tokens in a Hansard transcript line that are part of a speaker
    name."""

    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

    # @property
    # def source_vocab_size(self):
    #     return 2**8  # 256

    # @property
    # def targeted_vocab_size(self):
    #     return 3

    @property
    def is_generate_per_split(self):
        return False

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


    @property
    def source_encoder(self):
        return text_encoder.ByteTextEncoder()
    

    @property
    def targets_encoder(self):
        return text_encoder.TokenTextEncoder(vocab_filename=None, vocab_list=CLASS_LABELS)


    def feature_encoders(self, data_dir):
        return {
            "inputs": self.source_encoder,
            "targets": self.targets_encoder,
        }


    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        is_train = dataset_split == problem.DatasetSplit.TRAIN
        if is_train:
            dataset_path = DATA_PATH
        else:
            dataset_path = DATA_PATH
        lines = pd.read_csv(dataset_path)
        assert lines.text.isnull().sum() == 0, "Every line should have a non-null text value."
        # removes lines without a speaker name.
        lines = lines[lines.start.notnull() | lines.end.notnull()]
        # fills in missing start/end positions.
        lines.start[lines.start.isnull()] = 0
        lines.end[lines.end.isnull()] = lines.text[lines.end.isnull()].apply(lambda x: len(x))
        lines.start -= 1
        lines.end -= 1
        for _, line in lines.iterrows():
            line_text = line['text']
            start = int(line['start'])
            end = int(line['end'])
            if self.vocab_type == text_problems.VocabType.CHARACTER:
                targets = ['O'] * len(line_text)
                targets[start] = 'B-speaker'
                for i in range(start + 1, end + 1):
                    targets[i] = 'I-speaker'
            else:
                raise NotImplementedError()
            # targets.append('<eos>')
            yield {'inputs': line_text, 'targets': ' '.join(targets)}


    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        # encoder = self.get_or_create_vocab(data_dir, tmp_dir)
        return text_problems.text2text_generate_encoded(generator, vocab=self.source_encoder,
            targets_vocab=self.targets_encoder, has_inputs=self.has_inputs)
