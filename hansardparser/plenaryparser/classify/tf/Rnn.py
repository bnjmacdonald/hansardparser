
import tensorflow as tf
from hansardparser.plenaryparser.classify.tf import Classifier

class RnnClassifier(Classifier):
    """Implements a recurrent neural network classifier.

    """

    def __str__(self):
        return '{0}'.format(self.config.cell_type.lower()) if self.config.cell_type else 'rnn'

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
            self.inputs_placeholder = tf.placeholder(tf.int64, shape=[self.config.n_examples, self.config.max_seqlen], name='input')
            self.labels_placeholder = tf.placeholder(tf.int64, shape=[self.config.n_examples], name='labels')
            self.seqlens_placeholder = tf.placeholder(tf.int32, shape=[self.config.n_examples], name='seqlen')
            self.input_words = tf.Variable(self.inputs_placeholder, trainable=False, collections=[])
            self.input_labels = tf.Variable(self.labels_placeholder, trainable=False, collections=[])
            self.input_seqlens = tf.Variable(self.seqlens_placeholder, trainable=False, collections=[])
            single_input, single_label, single_seqlen = tf.train.slice_input_producer([self.input_words, self.input_labels, self.input_seqlens], num_epochs=self.config.n_epochs)
            # single_label = tf.cast(single_label, tf.int32)
            self.inputs, self.labels, self.seqlens = tf.train.batch([single_input, single_label, single_seqlen], batch_size=self.config.batch_size, allow_smaller_final_batch=True)  # dynamic_pad=True
            self.dropout_placeholder = tf.placeholder(tf.float32, name='dropout')
            # self.this_batch_size_placeholder = tf.placeholder(tf.int32, name='this_batch_size')

    def _add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input
        data into predictions.

        Returns:

            pred: A tensor of shape (batch_size, n_classes)
        """
        batch_size = tf.shape(self.inputs)[0]
        rnn_inputs = tf.nn.embedding_lookup(self.embed_matrix, self.inputs, name='embed')
        
        # adds layers.
        if self.config.cell_type == 'LSTM':
            stacked_cell = tf.nn.rnn_cell.MultiRNNCell([self._rnn_cell() for _ in range(self.config.n_layers)], state_is_tuple=True)
        else:
            stacked_cell = tf.nn.rnn_cell.MultiRNNCell([self._rnn_cell() for _ in range(self.config.n_layers)])
        init_state = stacked_cell.zero_state(batch_size, tf.float32)
        # init_state = tf.zeros([batch_size, lstm.state_size])
        rnn_outputs, final_state = tf.nn.dynamic_rnn(stacked_cell, rnn_inputs, sequence_length=self.seqlens, initial_state=init_state)
        with tf.variable_scope('softmax'):
            # xavier_initializer = xavier_weight_init()
            # W = xavier_initializer((self.config.n_hidden, self.config.n_classes), name='weights')
            W = tf.get_variable('weights', [self.config.n_hidden, self.config.n_classes])
            b = tf.Variable(tf.zeros([self.config.n_classes]), name='bias')
            # b = tf.get_variable('b', [self.config.n_classes], initializer=tf.constant_initializer(0.0))
        # reshape rnn_outputs
        # rnn_outputs = tf.reshape(rnn_outputs, [-1, self.config.n_hidden])
        last_rnn_output = tf.gather_nd(rnn_outputs, tf.stack([tf.range(batch_size), self.seqlens-1], axis=1))
        pred = tf.add(tf.matmul(last_rnn_output, W), b)
        return pred
    
    def _rnn_cell(self):
        """returns an RNN cell."""
        if self.config.cell_type == 'GRU':
            cell = tf.nn.rnn_cell.GRUCell(self.config.n_hidden)
        elif self.config.cell_type == 'LSTM':
            cell = tf.nn.rnn_cell.LSTMCell(self.config.n_hidden, state_is_tuple=True)
        else:
            cell = tf.nn.rnn_cell.BasicRNNCell(self.config.n_hidden)
        # adds dropout
        if self.config.use_dropout:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.dropout_placeholder)
        return cell
