
import os
import re
import json
import pandas as pd

import tensorflow as tf
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems, text_encoder
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
from . import config
from .utils import extract_flatworld_tags, get_line_context, normalize_text, split_lines


@registry.register_problem
class HansardLineType4(text_problems.Text2ClassProblem):
    f"""Predicts the type of each line in a Hansard transcript.
    
    Line types: {config.LINE_LABEL_CODES.keys()}.
    """

    CONTEXT_SEPARATOR = "<EOC>"
    CONTEXT_SEPARATOR_ID = 2

    @property
    def additional_reserved_tokens(self):
        return [self.CONTEXT_SEPARATOR]

    @property
    def vocab_type(self):
        return text_problems.VocabType.SUBWORD

    @property
    def approx_vocab_size(self):
        """Approximate vocab size to generate. Only for VocabType.SUBWORD."""
        return 2**16  # ~64k

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

    def feature_encoders(self, data_dir):
        encoder = self.get_or_create_vocab(data_dir, None, force_get=True)
        encoders = {'inputs': encoder, 'context': encoder}
        return encoders

    def generate_text_for_vocab(self, data_dir, tmp_dir):
        for i, sample in enumerate(self.generate_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN)):
            yield sample["inputs"]
            yield sample["context"]
            # yield sample["targets"]
            if self.max_samples_for_vocab and (i + 1) >= self.max_samples_for_vocab:
                break

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoder = self.get_or_create_vocab(data_dir, tmp_dir)
        encoders = {'inputs': encoder, 'context': encoder}
        # vocab = self.feature_encoders(data_dir)["context"]
        for sample in generator:
            encoded_sample = self.encode_example(sample, encoders)
            yield encoded_sample

    def encode_example(self, example, encoders):
        """encodes an example.
        
        example is a dict containing the inputs (str), context (str), and label (int).
        """
        inputs = encoders['inputs'].encode(example["inputs"])
        inputs.append(text_encoder.EOS_ID)
        context = encoders['context'].encode(example["context"])
        context.append(text_encoder.EOS_ID)
        label = example["label"]
        return {"inputs": inputs, "context": context, "targets": [label]}

    def hparams(self, defaults, unused_model_hparams):
        (super(HansardLineType4, self)
         .hparams(defaults, unused_model_hparams))
        p = defaults
        p.modality["context"] = modalities.ModalityType.SYMBOL
        p.vocab_size["context"] = self._encoders["context"].vocab_size
        if self.packed_length:
            raise NotImplementedError("HansardLineType4 does not "
                                      "support packed_length")

    def example_reading_spec(self):
        data_fields, data_items_to_decoders = (super().example_reading_spec())
        data_fields["context"] = tf.VarLenFeature(tf.int64)
        return (data_fields, data_items_to_decoders)

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

    def preprocess_example(self, example, mode, hparams):
        sep = tf.convert_to_tensor([self.CONTEXT_SEPARATOR_ID],
                                    dtype=example["inputs"].dtype)
        example["inputs"] = tf.concat([example["inputs"][:-1], sep, example["context"]], 0)
        example = super().preprocess_example(example, mode, hparams)
        return example

    def _generate_csv_samples(self, dataset_path, is_train):
        lines = pd.read_csv(dataset_path)
        # keeps only lines where label exists and is not equal to 'blank'.
        lines = lines[(lines.label.notnull()) & (~lines.label.isin(['blank']))]
        # keeps lines without a note.
        lines = lines[lines.notes.isnull()]
        lines['text'] = lines.text.apply(normalize_text)
        assert lines.text.isnull().sum() == 0, "Every line must have a non-null text value."
        assert (lines.text.str.len() > 0).all(), "Every line must have a text value with len > 0."
        assert lines.label.isnull().sum() == 0, "Every line must have a non-null label."
        allowed_labels = set(config.LINE_LABEL_CODES.keys())
        assert len(set(lines.label.unique()).difference(allowed_labels)) == 0, \
            (f"Training data contains a non-allowable label. Labels in training "
             f"data: {lines.label.unique().tolist()}")
        assert len(allowed_labels.difference(set(lines.label.unique()))) == 0, \
            ("One or more labels does not appear in training data.")
        if is_train:
            # duplicates data, without `garbage` lines.
            lines_wo_garbage = lines[~lines.label.isin(['garbage'])]
            lines_wo_garbage['file'] += f'_wo_garbage'
            lines = pd.concat([lines, lines_wo_garbage], axis=0, ignore_index=True, sort=True)
            if config.N_SPLIT_LINES_PASSES > 0:
                # randomly splits lines to create more training data.
                lines_split = []
                for i in range(config.N_SPLIT_LINES_PASSES):
                    lines_split_temp = split_lines(lines)
                    lines_split_temp['file'] += f'_{i}'
                    lines_split.append(lines_split_temp)
                lines_split = pd.concat(lines_split, axis=0, ignore_index=True, sort=True)
                # idx = np.random.randint(low=0, high=lines_split.shape[0])
                # lines_split.iloc[idx:idx+10,:]
                lines = pd.concat([lines, lines_split], axis=0, ignore_index=True, sort=True)
            lines.sort_values(by=['year', 'file', 'line'], inplace=True)
        # retrieves context surrounding each line.
        contexts = get_line_context(lines, n=config.CONTEXT_N_LINES)
        assert contexts.prev_context.isnull().sum() == 0
        assert contexts.next_context.isnull().sum() == 0
        lines = pd.merge(lines, contexts, on=['year', 'file', 'line'], how='left', validate='1:1')
        # drops lines that have a null prev_context and are not the first N lines.
        lines = lines[~((lines.prev_context.str.len() == 0) & (lines.line > config.CONTEXT_N_LINES))]
        assert (lines[lines.line == 1].prev_context.str.len() == 0).all()
        # drops lines that have a null next_context and are not an end-of-document line.
        lines = lines[~((lines.next_context.str.len() == 0) & ~lines.text.str.contains(r'<EOD>$', flags=re.IGNORECASE))]
        lines[lines.text.str.contains(r'<EOD>$', flags=re.IGNORECASE)]['next_context'] = ''
        # lines['line_length'] = lines.text.str.len()
        # lines['line_context_length'] = lines.text.str.len() + lines.prev_context.str.len() + lines.next_context.str.len()
        # lines.line_length.quantile(q=pd.np.arange(0, 1.1, 0.1))
        # lines.line_context_length.quantile(q=pd.np.arange(0, 1.1, 0.1))
        if is_train and config.UPSAMPLE:
            lines['label_prop'] = lines.groupby('label').label.transform(lambda x: x.shape[0] / float(lines.shape[0]))
            lines['weight'] = lines.shape[0] / lines['label_prop']
            lines['weight'] /= lines['weight'].sum()
            assert lines['weight'].isnull().sum() == 0
            # sanity checks:
            # pd.crosstab(lines['label'], lines['label_prop'])
            # lines['label'].value_counts() / lines.shape[0]
            # pd.crosstab(lines['label'], lines['weight'])
            # samples 10x the number of examples in `lines`.
            lines = lines.sample(n=lines.shape[0] * config.UPSAMPLE_FACTOR, replace=True, weights=lines.weight.values, random_state=config.SEED)
            # sanity check: number of examples per label class should be roughly equal.
            assert all(abs(lines['label'].value_counts() / lines.shape[0] - 1.0 / lines['label'].nunique()) < 0.01), \
                ('There should be roughly equal numbers of examples in each label class after upsampling.')
        # lines = [{'text': l['text'], '_id': f"{l['_id']}-{l['line']}"} for ix, l in lines.iterrows()]
        assert lines.label.isnull().sum() == 0
        for _, line in lines.iterrows():
            line_text = line['text']
            prev_context = line['prev_context'] if pd.notnull(line['prev_context']) else ''
            next_context = line['next_context'] if pd.notnull(line['next_context']) else ''
            if config.RM_FLATWORLD_TAGS:
                # removes Flatworld tags from lines.
                line_text, _ = extract_flatworld_tags(line_text)
                prev_context, _ = extract_flatworld_tags(prev_context)
                next_context, _ = extract_flatworld_tags(next_context)
            context = '\n'.join([prev_context, next_context])
            label = config.LINE_LABEL_CODES[line['label']]
            yield {'inputs': line_text, 'context': context, 'label': label}


@registry.register_problem
class HansardLineType4Char(HansardLineType4):
    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER
