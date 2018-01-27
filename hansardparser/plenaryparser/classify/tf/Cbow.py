"""
"""

import tensorflow as tf
from hansardparser.plenaryparser.classify.tf import Classifier

class CbowClassifier(Classifier):
    """Implements a continuous bag-of-words feed forward classifier.

    Input layer is the sum of word embeddings for all tokens in a document.

    Output layer is a one-hot vector for each of the possible labels.
    
    The network predicts who the speaking person is based on this summation of the
    embeddings of all words in the speech. The person embeddings of interest
    are stored in the weights matrix.

    Usage:

        Example::

            >>> from hansardparser.plenaryparser.classify.tf.cbow import CbowClassifier
    """

    def __str__(self):
        return 'cbow'

    def _add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.

        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used as
        inputs by the rest of the model building and will be fed data during
        training.

        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders


        """
        with tf.name_scope('data'):
            # self.batch_size_placeholder = tf.placeholder(tf.int32, shape=(1,), name='batch_size')
            # batch_size = tf.gather(self.batch_size_placeholder, 0)
            # self.seqlens_placeholder = tf.placeholder(tf.int32, shape=[None], name='seqlen')
            self.inputs_placeholder = tf.placeholder(tf.int64, shape=[self.config.n_examples, self.config.max_seqlen], name='input')
            self.labels_placeholder = tf.placeholder(tf.int64, shape=[self.config.n_examples], name='labels')
            self.input_words = tf.Variable(self.inputs_placeholder, trainable=False, collections=[])
            self.input_labels = tf.Variable(self.labels_placeholder, trainable=False, collections=[])
            single_input, single_label = tf.train.slice_input_producer([self.input_words, self.input_labels], num_epochs=self.config.n_epochs)
            # single_label = tf.cast(single_label, tf.int32)
            self.inputs, self.labels = tf.train.batch([single_input, single_label], batch_size=self.config.batch_size, allow_smaller_final_batch=True)  # dynamic_pad=True

    def _add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input
        data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        x = tf.nn.embedding_lookup(self.embed_matrix, self.inputs, name='embed')
        with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE) as scope:
            # scope.reuse_variables()
            W = tf.get_variable('weights', [self.config.embed_size, self.config.n_classes], initializer=tf.contrib.layers.xavier_initializer())
            # b = tf.get_variable('bias', [self.config.n_classes], initializer=tf.constant_initializer(0.0))
            # b = tf.Variable(tf.zeros([self.config.n_classes]), name='bias')
            cbow = tf.reshape(tf.reduce_sum(x, axis=1), shape=[-1, self.config.embed_size], name='cbow')
        pred = tf.matmul(cbow, W)
        return pred



