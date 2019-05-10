"""Implements minore changes to tensor2tensor `text_cnn` model so that it
trains and decodes without errors.

Todos:

  TODO: implement multiple layers. 
  TODO: implement hparams for pool stride, pool filter size, and other parameters.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


@registry.register_model
class TextCNNV2(t2t_model.T2TModel):
  """Text CNN."""

  def body(self, features):
    """TextCNN main model_fn.
    
    This is exactly the same as TextCNN, except that tf.nn.conv2d has been replaced
    with tf.layers.conv2d in order to avoid the control-flow ValueError during
    decoding/export.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Text inputs.
              [batch_size, input_length, 1, hidden_dim].
          "targets": Target encoder outputs.
              [batch_size, 1, 1, hidden_dim]
    
    Returns:
      Final encoder representation. [batch_size, 1, 1, hidden_dim] # NOTE: this is out of date.
    """
    hparams = self._hparams
    inputs = features["inputs"]

    xshape = common_layers.shape_list(inputs)

    vocab_size = xshape[3]
    inputs = tf.reshape(inputs, [xshape[0], xshape[1], xshape[3], xshape[2]])
    if hparams.num_hidden_layers > 1:
      tf.logging.warn('TextCNNV2 with multiple layers is not implemented. Setting hparams.num_hidden_layers to 1.')
      hparams.num_hidden_layers = 1
    pooled_outputs = []
    for _, filter_size in enumerate(hparams.filter_sizes):
      with tf.name_scope("conv-maxpool-%s" % filter_size):
        # filter_shape = [filter_size, vocab_size, 1, hparams.num_filters]
        # filter_var = tf.Variable(
        #     tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        # filter_bias = tf.Variable(
        #     tf.constant(0.1, shape=[hparams.num_filters]), name="b")
        # conv = tf.nn.conv2d(
        #     inputs,
        #     filter_var,
        #     strides=[1, 1, 1, 1],
        #     padding="VALID",
        #     name="conv")
        # conv_outputs = tf.nn.relu(
        #     tf.nn.bias_add(conv, filter_bias), name="relu")
        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=hparams.num_filters,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_size=filter_size,
            strides=1,
            padding="VALID",
            activation=tf.nn.relu,
            use_bias=True,
            name="conv"
        )
        # conv_outputs = tf.nn.relu(conv, name="relu")
        # pooled = tf.math.reduce_max(
        #     conv_outputs, axis=1, keepdims=True, name="max")
        pooled = tf.layers.max_pooling2d(
          inputs=conv,
          pool_size=2,
          strides=2,
          name='pool'
        )
        pooled = tf.nn.dropout(pooled, 1 - hparams.output_dropout)
        pooled_outputs.append(pooled)
    num_filters_total = hparams.num_filters * len(hparams.filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    # output = tf.nn.dropout(h_pool, 1 - hparams.output_dropout)
    output = h_pool
    # h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # Add dropout
    # output = tf.nn.dropout(h_pool_flat, 1 - hparams.output_dropout)
    # output = tf.reshape(output, [-1, 1, 1, num_filters_total])
    # print(inputs.get_shape())
    # print(num_filters_total)
    # print(pooled.get_shape())
    # print(h_pool.get_shape())
    # print(h_pool_flat.get_shape())
    # print(output.get_shape())
    # original:
    # (?, ?, 64, 1)
    # 128
    # (?, 1, 1, 32)
    # (?, 1, 1, 128)
    # (?, 128)
    # (?, 1, 1, 128)
    # mine:
    # (?, ?, 64, 1)
    # 128
    # (?, 1, 63, 32)
    # (?, 1, 63, 128)
    # (?, 128)
    # (?, 1, 1, 128)
    return output


@registry.register_hparams
def text_cnn_base_v2():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 4096
  hparams.max_length = 256
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_schedule = "legacy"
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer_gain = 1.0
  hparams.num_hidden_layers = 1
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.num_sampled_classes = 0
  hparams.label_smoothing = 0.1
  hparams.shared_embedding_and_softmax_weights = True
  hparams.symbol_modality_num_shards = 16

  # Add new ones like this.
  hparams.add_hparam("filter_sizes", [2, 3, 4, 5])
  hparams.add_hparam("pool_size", 2)
  hparams.add_hparam("pool_stride", 2)
  hparams.add_hparam("num_filters", 128)
  hparams.add_hparam("output_dropout", 0.4)
  return hparams
