"""

Notes:

    * to generate an input dataset, use `hansardparser.plenaryparser.build_training_set.mk_lines_corpus`.

Usage:

    Simple feed-forward CBOW classifier::

        python -m hansardparser.plenaryparser.classify.tf.train -m \
            --verbosity 2 --train \
            --classifier cbow \
            --corpus data/generated/text2vec/corpora/2018-01-27T013828 \
            --builder data/generated/text2vec/builders/2018-01-27T013828 \
            --batch_size 1000 \
            --embed_size 100 \
            --n_epochs 10
    
    RNN classifier with basic RNN cell::
        
        python -m hansardparser.plenaryparser.classify.tf.train -m \
            --verbosity 2 --train \
            --classifier rnn \
            --outpath experiments/tf/model$(ls experiments/tf | wc -l | tr -d " ") \
            --corpus data/generated/text2vec/corpora/2018-01-27T013828 \
            --builder data/generated/text2vec/builders/2018-01-27T013828 \
            --batch_size 1000 \
            --cell_type rnn \
            --embed_size 100 \
            --n_epochs 10
    
    RNN classifier with LSTM cell::
        
        python -m hansardparser.plenaryparser.classify.tf.train -m \
            --verbosity 2 --train \
            --classifier rnn \
            --outpath experiments/tf/model$(ls experiments/tf | wc -l | tr -d " ") \
            --corpus data/generated/text2vec/corpora/2018-01-27T013828 \
            --builder data/generated/text2vec/builders/2018-01-27T013828 \
            --batch_size 1000 \
            --cell_type lstm \
            --embed_size 100 \
            --n_epochs 10

Todos:

    TODO: implement use of tf.data API to stream data from disk, rather than
        loading all into memory.
    
    TODO: 
"""

import os
import sys
import argparse
from typing import List, Tuple
import numpy as np
import tensorflow as tf

from text2vec.corpora.corpora import Corpus, CorpusBuilder
from hansardparser.plenaryparser.classify.tf import CbowClassifier, RnnClassifier
from hansardparser.plenaryparser.classify.tf.config import ClassifierConfig
from hansardparser.plenaryparser.classify.utils import parse_unknown_args

import settings

CLASSIFIERS = {
    'cbow': CbowClassifier,
    'rnn': RnnClassifier
}

def parse_args(args: List[str] = None, validate: bool = True) -> Tuple[argparse.Namespace, argparse.Namespace]:
    """parses command-line arguments for the `train.py` module."""
    parser = argparse.ArgumentParser(add_help=True)
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    parser.add_argument('--corpus', type=str, help="corpus path")
    parser.add_argument('--builder', type=str, help="corpus builder path")
    parser.add_argument('--classifier', type=str, help="classifier to train", choices=['cbow', 'rnn'])
    parser.add_argument('--train', action='store_true', default=False, help='Train model (if False, only evaluate on dev set).')
    parser.add_argument('--outpath', type=str, help="path to directory where model should be saved (if model already exists in this diretory, it will be restored).")
    args, unknown = parser.parse_known_args(args)
    config_args = parse_unknown_args(unknown)
    sys.argv = [sys.argv[0]]  # kludge.
    if validate:
        if not args.train and args.outpath is None:
            raise RuntimeError('If --train flag is not given, then an outpath must be provided (--outpath)')
    return args, config_args


def main(args: argparse.Namespace = None, config_args: argparse.Namespace = None) -> None:
    if args is None:
        args = argparse.Namespace()
    if config_args is None:
        config_args = argparse.Namespace()
    config = ClassifierConfig(verbosity=args.verbosity, **config_args.__dict__)
    if config.debug:
        print('RUNNING IN DEBUG MODE')
        config.n_epochs = 2
    if not args.train:
        config.n_epochs = 1
    # loads data for evaluation.
    X_eval, y_eval, seqlens_eval, dictionary_eval = load_data(args.corpus, args.builder, split='dev', classifier=args.classifier, verbosity=args.verbosity)
    # loads training data.
    if args.train:
        X, y, seqlens, dictionary = load_data(args.corpus, args.builder, split='train', classifier=args.classifier, verbosity=args.verbosity)
        # updates config
        config.max_seqlen = min(config.max_seqlen, X.shape[1])
        config.n_examples = len(X)
        config.n_inputs = len(dictionary.token2id)  # vocab size
        config.n_classes = np.unique(y).shape[0]
        # KLUDGE: ensures that each example in X_eval has the same seqlen (i.e.
        # same number of features) as each example in X.
        if X_eval.shape[1] > config.max_seqlen:
            X_eval = X_eval[:,:config.max_seqlen]
        elif X_eval.shape[1] < config.max_seqlen:
            pad = 0
            X_new = np.full((X_eval.shape[0], config.max_seqlen - X_eval.shape[1]), pad)
            X_eval = np.hstack([X_eval, X_new])
    assert X.shape[1] == X_eval.shape[1], 'train set and dev set must have same number of features.'
    tf.reset_default_graph()
    clf = CLASSIFIERS[args.classifier](config=config, outpath=args.outpath)
    clf.build()
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        saver = tf.train.Saver(max_to_keep=2)
        # if checkpoint exists, restore from checkpoint
        clf.restore(sess, saver)
        writer_eval = tf.summary.FileWriter(os.path.join(args.outpath, 'eval'))
        if args.train:
            writer_train = tf.summary.FileWriter(os.path.join(args.outpath, 'train'), sess.graph)
            clf.train(
                X=X,
                y=y,
                seqlens=seqlens,
                sess=sess,
                saver=saver,
                writer=writer_train,
                pretrained_embeddings=None,
                evaluate_kws={
                    'inputs': X_eval,
                    'labels': y_eval,
                    'seqlens': seqlens_eval,
                    'writer': writer_eval,
                    # 'pretrained_embeddings': None
                }
            )
        else:
            clf.evaluate(x=X_eval, y=y_eval, seqlens=seqlens_eval, sess=sess, writer=writer_eval)
    return None


def load_data(corpus_path: str, builder_path: str, split: str, classifier: str, verbosity: int = 0):
    """loads data."""
    # loads corpus builder and corpus.
    if verbosity > 0:
        print('loading {0} data...'.format(split))
    builder = CorpusBuilder(os.path.join(builder_path, split))
    corpus = Corpus(os.path.join(corpus_path, split), builder=builder)
    # KLUDGE: loads all data into memory rather than reading from disk.
    corpus = np.array([doc for doc in corpus])
    labels = np.loadtxt(os.path.join(corpus_path, split, 'labels.txt'), dtype=np.int64)
    assert corpus.shape[0] == labels.shape[0], 'corpus and labels must have same number of observations.'
    seqlens = None
    if classifier == 'rnn':
        seqlens = np.loadtxt(os.path.join(corpus_path, split, 'seqlens.txt'), dtype=np.int64)
        assert corpus.shape[0] == seqlens.shape[0], 'corpus and seqlens must have same number of observations.'
    return corpus, labels, seqlens, builder.dictionary


if __name__ == '__main__':
    main()
