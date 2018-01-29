"""tests for RnnClassifier.
"""

import unittest
import os
import shutil
import argparse
import traceback
import numpy as np
import tensorflow as tf
from hansardparser.plenaryparser.classify.tf.config import ClassifierConfig
from hansardparser.plenaryparser.classify.tf import RnnClassifier
from hansardparser.plenaryparser.classify.tf import train
from hansardparser import settings

class RnnClassifierTests(unittest.TestCase):
    """tests for RnnClassifier.
    """

    def test_build(self):
        """tests that classifier builds without raising an exception.
        """
        try:
            tf.reset_default_graph()
            clf = RnnClassifier(config=ClassifierConfig(), outpath=None)
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
        """tests RnnClassifier().train() method.

        Todos:

            TODO: make this test faster by using smaller training and dev sets.
        """
        outpath = 'experiments/tests/rnn_classifier/'
        if os.path.exists(outpath):
            shutil.rmtree(outpath)
        os.makedirs(outpath)
        args = argparse.Namespace(
            verbosity=2,
            classifier='rnn',
            train=True,
            corpus=os.path.join(settings.DATA_ROOT, 'tests', 'classify', 'tf', 'corpora', 'one_hansard'),
            builder=os.path.join(settings.DATA_ROOT, 'tests', 'classify', 'tf', 'builders', 'one_hansard'),
            outpath=outpath
        )
        config_args = argparse.Namespace(
            n_epochs=2,
            batch_size=100,
            embed_size=50,
            eval_every=50,
            n_layers=2,
            n_hidden=75,
            cell_type='lstm'
        )
        try:
            train.main(args, config_args)
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.fail('train.main encountered an exception.')


if __name__ == '__main__':
    unittest.main()
