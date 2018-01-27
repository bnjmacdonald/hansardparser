"""configuration for a tf classifier.
"""

from typing import List

class ClassifierConfig(object):
    """configuration for a tf classifier.

    Attributes:

        batch_size: int = 1000. Size of each batch of examples for training.

        cell_type: str = 'rnn'. Type of RNN cell.

        clip_gradients: bool = False. Clip gradients.

        debug: bool = False. Debugging mode.

        embed_size: int = 100. Number of embedding dimensions in input
            embedding matrix.

        eval_every: int = 100. Evaluate classifier performance every N batches.

        learning_rate: float = 0.01. Learning rate.

        max_grad_norm: float = 5. Maximum gradient norm at which to clip
            gradients. Does nothing if clip_gradients==False.

        max_seqlen: int = 200. Max sequence length. Only relevant to RNN models.

        n_classes: int = 2. Number of label classes for output prediction.

        n_examples: int = 10000. Total number of examples in input data.

        n_inputs: int = 10000. Number of unique feature values (e.g. vocab size).

        n_epochs: int = 5. Number of training epochs over whole training set.

        n_hidden: int = 100. Number of neurons in each hidden layer.

        n_layers: int = 1. Number of hidden layers.

        p_keep: float = 0.5. Probability a neuron is kept. Does nothing if
            use_dropout==False.

        save_every: int = 100. Save after every N training steps.

        use_dropout: bool = False. Use neuron dropout for regularization.

        use_pretrained: bool = False. Use pretrained embeddings.
    """
    def __init__(self,
                 batch_size: int = 1000,
                 cell_type: str = 'rnn',
                 clip_gradients: bool = False,
                 debug: bool = False,
                 embed_size: int = 100,
                 eval_every: int = 100,
                 learning_rate: float = 0.01,
                 max_grad_norm: float = 5.,
                 max_seqlen: int = 200,
                 n_classes: int = 2,
                 n_examples: int = 10000,
                 n_inputs: int = 10000,
                 n_epochs: int = 5,
                 n_hidden: int = 100,
                 n_layers: int = 1,
                 p_keep: float = 0.5,
                 save_every: int = 100,
                 use_dropout: bool = False,
                 use_pretrained: bool = False,
                 verbosity: int = 0,
                 **kwargs
                 ) -> None:
        self.batch_size = batch_size
        self.cell_type = cell_type
        self.clip_gradients = clip_gradients
        self.debug = debug
        self.embed_size = embed_size
        self.eval_every = eval_every
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.max_seqlen = max_seqlen
        self.n_classes = n_classes
        self.n_examples = n_examples
        self.n_inputs = n_inputs
        self.n_epochs = n_epochs
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.p_keep = p_keep
        self.save_every = save_every
        self.use_dropout = use_dropout
        self.use_pretrained = use_pretrained
        self.verbosity = verbosity
        # self.shuffle = True
