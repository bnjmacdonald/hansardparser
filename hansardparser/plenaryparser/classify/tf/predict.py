"""predicts string matches from a trained siamese neural network.

Todos:

    TODO: implement data loading?
"""

import os
import warnings
import json
import sys
import argparse
from typing import List, Tuple
import numpy as np
import tensorflow as tf

from tftxtclassify.classifiers import TextClassifier
from tftxtclassify.utils import resize_sequences
from hansardparser.plenaryparser.classify.utils import parse_unknown_args
from hansardparser.plenaryparser.build_training_set.utils import build_temp_corpus
from hansardparser.plenaryparser.classify.utils import batchify
from hansardparser import settings

# KLUDGE: so that builder can be loaded. Best solution is probably to do away with
# the corpus-builder intermediate logic altogether. Instead, do the preprocessing
# at training time. Then it is also easier to experiment with alternative preprocessing
# as hyper-parameters. Simple define an API that covers many of the most important
# preprocessing options. I should add the preprocessors to tftxtclassify and get
# rid of text2vec.
from text2vec.processing.preprocess import preprocess_one
from hansardparser.plenaryparser.build_training_set.utils import str2ascii_safe


# DEBUG_SIZE = 200
DEFAULT_BATCH_SIZE = 1000


def parse_args(args: List[str] = None) -> argparse.Namespace:
    """parses command-line arguments for the `train.py` module.

    Arguments:

        args: List[str] = None. List of command-line arguments defined in
            `parse_args`.

            Example::

                ["--debug", "--verbosity", "2"].

    Returns:

        args: argparse.Namespace. Command-line arguments defined in `parse_args`.

            Example::

                argparse.Namespace(
                    debug=True,
                    verbosity=2,
                    ...
                )
    """
    parser = argparse.ArgumentParser(add_help=True)
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('--outpath', type=str,
        help='path to directory where predictions should be saved.')
    parser.add_argument('--documents', type=str, dest="documents_path",
        help="path to file containing documents (one document per line).")
    parser.add_argument('--builder', type=str, dest="builder_path",
        help="corpus builder path")
    parser.add_argument('--clf', type=str, dest="clf_path",
        help="path to trained tensorflow classifier")
    parser.add_argument('--is_corpus', action='store_true', default=False,
        help="`--documents` argument points to a numeric corpus, not a file containing strings.")
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
        help="Number of predictions per batch.")
    # parser.add_argument('--debug', action='store_true', help='Debug mode.')
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    args = parser.parse_args(args)
    return args


def main(documents_path: str, outpath: str, is_corpus: bool = False, **kwargs) -> None:
    """reads in documents from a text file (one document per line) and predicts
    the class of each document.

    Predicted classes are saved to disk.

    Returns:

        None.
    """
    if is_corpus:
        raise NotImplementedError
        # TODO: load corpus inputs and seqlens.
        # pred_logits, pred_probs, pred_classes = predict_from_strings(inputs, **kwargs)
    else:
        with open(documents_path, 'r') as f:
            documents = f.readlines()
        if 'verbosity' in kwargs and kwargs['verbosity'] > 0:
            print(f'Loaded {len(documents)} documents.')    
        pred_logits, pred_probs, pred_classes = predict_from_strings(documents, **kwargs)
    np.savetxt(os.path.join(outpath, 'pred_logits.txt'), pred_logits, fmt='%f')
    np.savetxt(os.path.join(outpath, 'pred_probs.txt'), pred_probs, fmt='%f')
    np.savetxt(os.path.join(outpath, 'pred_classes.txt'), pred_classes, fmt='%f')
    return None


def predict(inputs: np.array,
            seqlens: np.array,
            clf_path: str,
            batch_size: int = DEFAULT_BATCH_SIZE,
            verbosity: int = 0) -> Tuple[np.array, np.array, np.array]:
    """predicts the class of an example using a trained tensorflow classifier.

    Restores the classifier from disk and invokes the classifier's `predict`
    method. This is a simple wrapper that makes it easier to predict values
    from a trained model without worrying about tensorflow particulars for
    restoring a model and managing a session.

    Recommended usage:

        * call `predict` as few times as possible, since each time it is called
            the tensorflow classifier graph must be restored (which takes a few
            seconds).

    Arguments:

        inputs: np.array with shape (n_examples, vocab_size). Array of
            inputs.

        seqlens: np.array with shape (n_examples, ). Array of sequence lengths
                for `inputs`.

        clf_path: str. Path to directory where trained tensorflow model exists.

        batch_size: int = 1000. Number of examples per batch.

    Returns:

        pred_logits, pred_probs, pred_classes: Tuple[np.array, np.array, np.array].

            pred_logits: np.array with shape (inputs.shape[0],). Each element
                represents the predicted value of each class for row i of
                `inputs`.

            pred_probs: np.array with shape (inputs.shape[0],). Each element
                represents the predicted probability of each class for row i
                of `inputs`.

            pred_classes: np.array with shape (inputs.shape[0],). Each element
                contains the predicted class of row i of `inputs`
    """
    tf.reset_default_graph()
    with tf.Session() as sess:
        clf = TextClassifier(
            sess=sess,
            verbosity=verbosity
        )
        # restores the classifier.
        clf.restore(path=clf_path)
        # pad = dictionary.token2id['<PAD>']
        pad = 0  # TODO: retrieve true padding from model.
        inputs = resize_sequences(inputs, max_seqlen=clf.config.n_features, pad=pad)
        # clips seqlens so that max < config.max_seqlen
        seqlens = np.clip(seqlens, 0, clf.config.n_features)
        pred_logits = []
        pred_probs = []
        pred_classes = []
        n_batches = np.ceil(inputs.shape[0] / batch_size).astype(np.int32)
        if verbosity > 0:
            print(f'predicting values for {inputs.shape[0]} input pairs...')
        for i, batch_ix in enumerate(batchify(range(0, len(inputs)), batch_size)):
            if verbosity > 1:
                print(f'predicting values for batch {i} of {n_batches}...', end='\r')
            batch_pred_logits, batch_pred_probs, batch_pred_classes = clf.predict(
                get_probs=True,
                get_classes=True,
                inputs=inputs[batch_ix],
                seqlens=seqlens[batch_ix],
            )
            pred_logits.append(batch_pred_logits)
            pred_probs.append(batch_pred_probs)
            pred_classes.append(batch_pred_classes)
        if pred_logits[0].ndim > 1:
            pred_logits = np.vstack(pred_logits)
        else:
            pred_logits = np.hstack(pred_logits)
        if pred_probs[0].ndim > 1:
            pred_probs = np.vstack(pred_probs)
        else:
            pred_probs = np.hstack(pred_probs)
        pred_classes = np.hstack(pred_classes)
    assert pred_logits.shape[0] == inputs.shape[0]
    assert pred_probs.shape[0] == inputs.shape[0]
    assert pred_classes.shape[0] == inputs.shape[0]
    return pred_logits, pred_probs, pred_classes


def predict_from_strings(documents: np.array,
                         builder_path: str = None,
                         verbosity: int = 0,
                         **kwargs) -> np.array:
    """given an array of documents, predicts the class of each document using a trained
    tensorflow classifier.

    This is a wrapper to `predict`. Use `predict_from_strings` when your data is
    still in string format. Use `predict` if you have already converted your data
    to numeric format.

    Arguments:

        documents: np.array.

        builder_path: str = None. Path to corpus builder.

        **kwargs: keyword arguments to pass to `predict`.

    Returns:

        pred_logits, pred_probs, pred_classes: Tuple[np.array, np.array, np.array].
            See `predict`.

    Todos:

        TODO: include builder path in classifier `config.json` so that it can
            be loaded automatically.
    """
    # if builder_path is None, retrieve it from config of trained classifier.
    if builder_path is None:
        with open(os.path.join(kwargs['path'], 'config.json'), 'r') as f:
            config = json.load(f)
            builder_path = config['corpus_builder']
    kwargs['verbosity'] = verbosity
    inputs, seqlens = build_temp_corpus(documents, builder_path)
    # make predictions
    pred_logits, pred_probs, pred_classes = predict(
        inputs=inputs,
        seqlens=seqlens,
        **kwargs
    )
    return pred_logits, pred_probs, pred_classes


if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)
