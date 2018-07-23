
import os
import re
import json
import warnings
import pandas as pd
import tensorflow as tf

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems, text_encoder, generator_utils
from tensor2tensor.utils import registry
from . import config
from .hansard_line_type4 import HansardLineType4
from .utils import extract_flatworld_tags


@registry.register_problem
class HansardLineType4Predict(HansardLineType4):
    """Predicts the type of each line in a Hansard transcript.

    This class is identical to `HansardLineType4Predict`, except that this class
    is meant to be used only for (a) generating unlabeled serialized TFRecord
    data files; and (b) serving the trained model and making predictions from
    unserialized strings.
    """

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TEST,
            "shards": 9,
        }]


    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoder = self.get_or_create_vocab(data_dir, tmp_dir)
        for sample in generator:
            inputs = encoder.encode(sample["inputs"])
            inputs.append(text_encoder.EOS_ID)
            # label = sample["label"]
            yield {"inputs": inputs}


    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        samples = self._generate_txt_samples(tmp_dir)
        for sample in samples:
            yield sample


    def _generate_txt_samples(self, dataset_path):
        filenames = os.listdir(dataset_path)
        for filename in filenames:
            if filename.startswith('.'):
                continue
            tf.logging.info(f'serializing {filename}')
            file_path = os.path.join(dataset_path, filename)
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if config.RM_FLATWORLD_TAGS:
                        line, _ = extract_flatworld_tags(line)
                    yield {'inputs': line}


    def feature_encoders(self, data_dir):
        encoder = self.get_or_create_vocab(data_dir, None, force_get=True)
        return {
            "inputs": encoder,
        }


    def example_reading_spec(self):
        data_fields = {
            "inputs": tf.VarLenFeature(tf.int64)
        }
        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)


    def decode_example(self, string_example):
        """Return a dict of Tensors from a string.
        
        Todos:

            TODO: this method is a total kludge. Figure out a better way of allowing
                for the input to be a non-serialized string.
        """
        data_fields, data_items_to_decoders = self.example_reading_spec()
        ex = generator_utils.to_example({'batch_prediction_key': [0]}).SerializeToString()
        # Necessary to rejoin examples in the correct order with the Cloud ML Engine
        # batch prediction API.
        data_fields["batch_prediction_key"] = tf.FixedLenFeature([1], tf.int64, 0)
        data_fields.pop('inputs')
        if data_items_to_decoders is None:
            data_items_to_decoders = {
                field: tf.contrib.slim.tfexample_decoder.Tensor(field)
                for field in data_fields
            }
        decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
            data_fields, data_items_to_decoders)
        decode_items = list(sorted(data_items_to_decoders))
        decoded = decoder.decode(ex, items=decode_items)
        decoded_example = dict(zip(decode_items, decoded))
        # converts string to array of ints.
        if self.vocab_type == text_problems.VocabType.CHARACTER:
            decoded_example['inputs'] = tf.decode_raw(string_example, out_type=tf.uint8)
        else:
            raise NotImplementedError()
        return decoded_example
