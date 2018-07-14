"""trains a single tensorflow line classifier.

Usage::

    - command line usage: `$ python -m hansardparser.plenaryparser.classify.tf.train ...`.

        - View available command line arguments: `$ python -m hansardparser.plenaryparser.classify.tf.train --help`

        - See `train_tf_classifier.sh` for an example.
    
    - python module usage::

        >>> from hansardparser.plenaryparser.classify.tf import train
        >>> train.main(outpath="OUTPATH",
                       corpus="PATH_TO_CORPUS",
                       builder="PATH_TO_CORPUS_BUILDER",
                       classifier="cnn",
                       n_examples=10000,
                       n_eval_examples=1000,
                       n_epochs=10,
                       learning_rate=0.01,
                       verbosity=2)

Notes:

    * to generate an input dataset, use the `hansardparser.plenaryparser.build_training_set` module.

Todos:

    TODO: replace `argparse` with `tf.flags`.

    TODO: use tf.data API to read data from disk, rather than reading all data
        into memory.
    
    TODO: add trainable non-zero initial state.

    TODO: use PCA to shrink pre-trained embeddings to smaller embed_size.
"""

import os
import sys
import argparse
import warnings
from typing import List, Tuple
import numpy as np
import tensorflow as tf

from text2vec.corpora.corpora import Corpus, CorpusBuilder
from text2vec.processing.preprocess import preprocess_one  # rm_stop_words_punct, rm_digits
from text2vec.corpora.dictionaries import BasicDictionary

from tftxtclassify.classifiers import TextClassifier
from tftxtclassify.classifiers.config import get_config_class
from tftxtclassify.utils import resize_sequences

from hansardparser.plenaryparser.classify.utils import parse_unknown_args
from hansardparser import settings

DEBUG_SIZE = 50
DEBUG_EPOCHS = 2


def parse_args(args: List[str] = None) -> Tuple[argparse.Namespace, argparse.Namespace]:
    """parses command-line arguments for the `train.py` module.

    Arguments:

        args: List[str] = None. List of command-line arguments defined in
            `parse_args`.

            Example::

                ["--debug", "--verbosity", "2", "--classifier", "siamese-rnn"].

    Returns:

        args, config_args: Tuple[argparse.Namespace, argparse.Namespace].
        
            args: argparse.Namespace. Command-line arguments defined in `parse_args`.

                Example::

                    argparse.Namespace(
                        debug=True,
                        verbosity=2,
                        classifier='siamese-rnn',
                        ...
                    )

            config_args: argparse.Namespace. Command-line arguments that define
                configuration of classifier to be trained.

                Example::

                    argparse.Namespace(
                        clip_gradients=True,
                        learning_rate=0.01,
                        n_epochs=2,
                        batch_size=100,
                        n_hidden=[100,100],
                        ...
                    )
    """
    parser = argparse.ArgumentParser(add_help=True)
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('--outpath', type=str,
        help='path to directory where model should be saved (if model already'
             'exists in this diretory, it will be restored).')
    parser.add_argument('--corpus', type=str, help="corpus path")
    parser.add_argument('--builder', type=str, help="corpus builder path")
    parser.add_argument('--classifier', type=str, help="classifier to train",
        choices=TextClassifier.ACCEPTABLE_CLASSIFIERS)
    parser.add_argument('--n_examples', type=int, default=None,
        help='Number of observations in train set to sample for training.'
             'Discards all other train data, such that an entire epoch consists only of this sample.'
             'Useful for debugging and large data sets. If not provided, entire training set is used.')
    parser.add_argument('--n_eval_examples', type=int, default=1000,
        help='Maximum number of observations in train/validation sets to sample'
             'for constructing final performance metric.')
    parser.add_argument('--restore', action='store_true',
        help='attempt to restore existing classifier and continue training, if '
             'classifier exists in `outpath`.')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    args, unknown = parser.parse_known_args(args)
    config_args = parse_unknown_args(unknown)
    # sys.argv = [sys.argv[0]]
    return args, config_args


def main(outpath: str,
         corpus: str,
         builder: str,
         classifier: str,
         n_examples: int = None,
         n_eval_examples: int = 1000,
         restore: bool = False,
         debug: bool = False,
         verbosity: int = 0,
         **kwargs) -> Tuple[dict, dict]:
    f"""trains a tensorflow string matching text classifier.

    This method is a wrapper that loads a dataset of string matches and non-matches,
    instantiates a classifier, and trains the classifier on this dataset.

    Arguments:

        corpus: str. Path to corpus.

            - KLUDGE: the `corpus` argument must point to a directory containing three folders:
                `manual_pos`, `semisup_neg', and `semisup_pos`. The first contains a
                corpus of manual name matches; the second contains a corpus of negative
                semi-supervised name matches; and the third contains a corpus of positive
                semi-supervised name matches. This is very kludgy, as it means that if
                additional folders are added or these folders are renamed then changes
                need to be made to this module.
        
        builder: str. Path to corpus builder.
        
        classifier: str. Type of classifier. Must be one of: {TextClassifier.ACCEPTABLE_CLASSIFIERS}.
        
        outpath: str. Path to where train results will be saved.
        
        n_examples: int = None. Number of observations in train set to sample for training.
            Discards all other train data, such that an entire epoch consists only of this sample.
            Useful for debugging and large data sets. If None (default), entire training set is used.
        
        n_eval_examples: int = 1000. Maximum number of observations in train/
            validation sets to sample for constructing final performance metric.

        restore: bool = False. If True, attempts to restore existing classifier
            and continue training, if classifier exists in `outpath`.

        debug: bool = False. Execute in debug mode.
        
        **kwargs: keyword arguments containing model configuration settings. Example::

            {{
                "n_epochs": 10,
                "learning_rate": 0.01,
                ...
            }}

    Returns:

        perf_train, perf_val: Tuple[dict, dict].

            perf_train: dict. Dict containing performance on a sample of the
                training set after the classifier has finished training.

            perf_valdation: dict. Dict containing performance on a sample of the
                validation set after the classifier has finished training.
    
    Example::

        >>> from services.string_match.tf import train
        >>> train.main(outpath="OUTPATH",
                       corpus="PATH_TO_CORPUS",
                       builder="PATH_TO_CORPUS_BUILDER",
                       classifier="siamese-cnn",
                       n_examples=10000,
                       n_eval_examples=1000,
                       n_epochs=10,
                       learning_rate=0.01,
                       verbosity=2)
    """
    assert classifier in TextClassifier.ACCEPTABLE_CLASSIFIERS, \
        f'`classifier` is {classifier}, but must be one of {TextClassifier.ACCEPTABLE_CLASSIFIERS}.'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    if debug:
        print('TRAINING IN DEBUG MODE.')
        kwargs['n_epochs'] = 2
        n_examples = DEBUG_SIZE
        n_eval_examples = DEBUG_SIZE
    # loads data for validation set.
    inputs_val, labels_val, seqlens_val, _ = load_data(
        corpus_path=corpus,
        builder_path=builder,
        split='dev',
        classifier=classifier,
        verbosity=verbosity
    )
    # loads training data.
    inputs, labels, seqlens, dictionary = load_data(
        corpus_path=corpus,
        builder_path=builder,
        split='train',
        classifier=classifier,
        verbosity=verbosity
    )
    vocabulary = np.array([dictionary.id2token[i] for i in range(len(dictionary.id2token))])
    # if `n_examples`, sample a subset of the training data.
    if n_examples is not None:
        indices = np.random.choice(np.arange(0, labels.shape[0]), size=min(n_examples, labels.shape[0]), replace=False)
        inputs = inputs[indices]
        seqlens = seqlens[indices]
        labels = labels[indices]
    # initializes config
    if 'n_features' not in kwargs:
        kwargs['n_features'] = inputs.shape[1]
    else:
        # reduces `config.n_features` if inputs has fewer than `config.n_features``
        kwargs['n_features'] = min(kwargs['n_features'], inputs.shape[1])
    config = get_config_class(classifier)(
        classifier=classifier,
        outpath=outpath,
        n_examples=inputs.shape[0],
        n_classes=np.unique(labels).shape[0],
        class_weights=labels.shape[0] / np.bincount(labels),
        vocab_size=vocabulary.shape[0],
        **kwargs
    )
    config.corpus_builder = builder  # so that builder path is stored with model config.
    # truncates each input if greater than n_features.
    pad = dictionary.token2id['<PAD>']
    inputs = resize_sequences(inputs, max_seqlen=config.n_features, pad=pad)
    # KLUDGE: ensures that each example in inputs_val has the same seqlen (i.e.
    # same number of features) as each example in X.
    inputs_val = resize_sequences(inputs_val, max_seqlen=config.n_features, pad=pad)
    assert inputs.shape[1] == inputs_val.shape[1], 'train set and validation set must have same number of features.'
    # clips seqlens so that max < config.n_features
    seqlens = np.clip(seqlens, 0, config.n_features)
    seqlens_val = np.clip(seqlens_val, 0, config.n_features)
    # loads pretrained character embeddings.
    pretrained_embeddings = None
    if config.use_pretrained:
        raise NotImplementedError
        # TODO: implement with pretrained character embeddings. Train embeddings on Hansards.
        # pretrained_embeddings = load_char_embeddings(dictionary.id2token)
        # if hasattr(config, 'embed_size') and config.embed_size != pretrained_embeddings.shape[1]:
        #     warnings.warn(f'embed_size {config.embed_size} was given in `config_args`, '
        #                   f'but is being set to {pretrained_embeddings.shape[1]} to match '
        #                    'pretrained_embeddings.', RuntimeWarning)
        # config.embed_size = pretrained_embeddings.shape[1]
    tf.reset_default_graph()
    session_kws = {}
    if verbosity > 2:
        # logs device placement of every op
        session_kws['config'] = tf.ConfigProto(log_device_placement=True)
    with tf.Session(**session_kws) as sess:
        # initializes classifier and builds computational graph.
        clf = TextClassifier(
            sess=sess,
            config=config,
            vocabulary=vocabulary,
            verbosity=verbosity
        )
        # restores existing classifier and its graph, if exists.
        if restore:
            clf.restore()
        elif os.path.isfile(os.path.join(outpath, 'checkpoint')):
            warnings.warn(f'`checkpoint` file exists in {outpath}, but `restore=False`. '
                           'Existing classifier will be overwritten.', RuntimeWarning)
        # if classifier was not restored, then build graph now.
        if not clf._built:
            clf.build_graph()
        # trains the classifier.
        writer = tf.summary.FileWriter(outpath, sess.graph)
        clf.train(
            inputs=inputs,
            seqlens=seqlens,
            labels=labels,
            writer=writer,
            pretrained_embeddings=pretrained_embeddings,
            validation_kws={
                'inputs': inputs_val,
                'labels': labels_val,
                'seqlens': seqlens_val,
                'pretrained_embeddings': pretrained_embeddings
            }
        )
        if verbosity > 0:
            print('\nTraining has finished. Final performance results:')
        # final performance on sample of training set.
        # KLUDGE: this implementation is kludgy, as _predict_batch and _evaluate_batch
        # do other things internally that probably don't need to be done.
        train_indices = np.random.choice(np.arange(0, inputs.shape[0]),
            size=min(n_eval_examples, inputs.shape[0]), replace=False)
        preds_train, _, _ = clf.predict(
            get_probs=False,
            get_classes=False,
            inputs=inputs[train_indices],
            seqlens=seqlens[train_indices]
        )
        perf_train = clf._evaluate_batch(
            pred_logits=preds_train,
            labels=labels[train_indices],
            inputs=inputs[train_indices],
            seqlens=seqlens[train_indices],
            is_validation=False
        )
        # final performance on sample of validation set.
        val_indices = np.random.choice(np.arange(0, inputs_val.shape[0]),
            size=min(n_eval_examples, inputs_val.shape[0]), replace=False)
        preds_val, _, _ = clf.predict(
            get_probs=False,
            get_classes=False,
            inputs=inputs_val[val_indices],
            seqlens=seqlens_val[val_indices]
        )
        perf_val = clf._evaluate_batch(
            pred_logits=preds_val,
            labels=labels_val[val_indices],
            inputs=inputs_val[val_indices],
            seqlens=seqlens_val[val_indices],
            is_validation=True
        )
        writer.close()
    return perf_train, perf_val


def load_data(corpus_path: str,
              builder_path: str,
              split: str,
              classifier: str,
              verbosity: int = 0) -> Tuple[np.array, np.array, np.array, BasicDictionary]:
    """loads data."""
    # loads corpus builder and corpus.
    if verbosity > 0:
        print(f'loading {split} data...')
    builder = CorpusBuilder(builder_path)
    corpus = Corpus(os.path.join(corpus_path, split), builder=builder)
    # KLUDGE: loads all data into memory rather than reading from disk.
    corpus = np.array([doc for doc in corpus])
    labels = np.loadtxt(os.path.join(corpus_path, split, 'labels.txt'), dtype=np.int64)
    assert corpus.shape[0] == labels.shape[0], 'corpus and labels must have same number of observations.'
    # seqlens = None
    # if classifier == 'rnn':
    seqlens = np.loadtxt(os.path.join(corpus_path, split, 'seqlens.txt'), dtype=np.int64)
    assert corpus.shape[0] == seqlens.shape[0], 'corpus and seqlens must have same number of observations.'
    return corpus, labels, seqlens, builder.dictionary


if __name__ == '__main__':
    args, config_args = parse_args()
    kwargs = dict(**args.__dict__, **config_args.__dict__)
    main(**kwargs)
