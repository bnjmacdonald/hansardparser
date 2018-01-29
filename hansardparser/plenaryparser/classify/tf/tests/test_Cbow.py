"""tests for CbowClassifier.
"""

import unittest
import os
import shutil
import argparse
import traceback
import numpy as np
import tensorflow as tf
from hansardparser.plenaryparser.classify.tf.config import ClassifierConfig
from hansardparser.plenaryparser.classify.tf import CbowClassifier
from hansardparser.plenaryparser.classify.tf import train
from hansardparser import settings

class CbowClassifierTests(unittest.TestCase):
    """tests for CbowClassifier.
    """

    # @classmethod
    # def setUpClass(cls):
    #     config = ClassifierConfig(
    #     )
    #     corpus_path = os.path.join(settings.DATA_ROOT, 'tests', 'classify', 'tf', 'corpora', '2018-01-29T004652')
    #     builder_path = os.path.join(settings.DATA_ROOT, 'tests', 'classify', 'tf', 'builders', '2018-01-29T004652')
    #     classifier = 'Cbow'
    #     verbosity = 2
    #     # loads dev data.
    #     X_eval, y_eval, seqlens_eval, dictionary_eval = load_data(corpus_path, builder_path, split='dev', classifier=classifier, verbosity=verbosity)
    #     # loads training data.
    #     X, y, seqlens, dictionary = load_data(corpus_path, builder_path, split='train', classifier=classifier, verbosity=verbosity)
    #     # updates config
    #     config.max_seqlen = min(config.max_seqlen, X.shape[1])
    #     config.n_examples = len(X)
    #     config.n_inputs = len(dictionary.token2id)  # vocab size
    #     config.n_classes = np.unique(y).shape[0]
    #     # KLUDGE: ensures that each example in X_eval has the same seqlen (i.e.
    #     # same number of features) as each example in X.
    #     if X_eval.shape[1] > config.max_seqlen:
    #         X_eval = X_eval[:,:config.max_seqlen]
    #     elif X_eval.shape[1] < config.max_seqlen:
    #         pad = 0
    #         X_new = np.full((X_eval.shape[0], config.max_seqlen - X_eval.shape[1]), pad)
    #         X_eval = np.hstack([X_eval, X_new])
    #     assert X.shape[1] == X_eval.shape[1], 'train set and dev set must have same number of features.'
    #     cls.outpath = 'test_output/'
    #     cls.config = config
    #     cls.X = X
    #     cls.y = y
    #     cls.seqlens = seqlens
    #     cls.X_eval = X_eval
    #     cls.y_eval = y_eval
    #     cls.seqlens_eval = seqlens_eval

    # def setUp(self):
    #     self.clf = CbowClassifier(config=self.config, outpath=self.outpath)

    def test_build(self):
        """tests that classifier builds without raising an exception.
        """
        try:
            tf.reset_default_graph()
            config = ClassifierConfig()
            clf = CbowClassifier(config=config, outpath=None)
            clf.build()
        except:
            self.fail('self.build() raised an error.')
        

    def test_build_network_config(self):
        """tests that classifier builds with appropriate network configuration.

        TODO: implement.
        """
        self.fail('test not implemented.')
    
    def test_build_train_config(self):
        """tests that classifier builds with appropriate training configuration
        (e.g. n_epochs, eval_every, ...).

        TODO: implement.
        """
        self.fail('test not implemented.')

    def test_train(self):
        """tests CbowClassifier().train() method.

        Todos:

            TODO: make this test faster by using smaller training and dev sets.
        """
        outpath = 'experiments/tests/cbow_classifier/'
        if os.path.exists(outpath):
            shutil.rmtree(outpath)
        os.makedirs(outpath)
        args = argparse.Namespace(
            verbosity=2,
            classifier='cbow',
            train=True,
            corpus=os.path.join(settings.DATA_ROOT, 'tests', 'classify', 'tf', 'corpora', 'one_hansard'),
            builder=os.path.join(settings.DATA_ROOT, 'tests', 'classify', 'tf', 'builders', 'one_hansard'),
            outpath=outpath
        )
        config_args = argparse.Namespace(
            n_epochs=2,
            batch_size=100,
            embed_size=50,
            eval_every=10,
            save_every=10,
            n_hidden=75,
        )
        try:
            train.main(args, config_args)
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.fail('train.main encountered an exception.')


if __name__ == '__main__':
    unittest.main()
