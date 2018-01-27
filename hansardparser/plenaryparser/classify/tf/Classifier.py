"""implements the `Model` class, which abstracts a Tensorflow graph for an
NLP classification task.
"""

import os
import traceback
import time
import tensorflow as tf
from sklearn import metrics
import numpy as np
from hansardparser.plenaryparser.classify.tf.utils import heatmap_confusion, Progbar


class Classifier(object):
    """Abstracts a Tensorflow graph for an NLP classification task.

    Built for classification. Uses embeddings for inputs.

    Exposes three main methods::

        `build`: builds the Tensorflow graph.

        `train`: trains a Tensorflow classifier.

        `evaluate`: evaluate the performance of a trained model.

    Attributes:

        config: dict. Model configuration, such as `n_epochs`, `embed_size`, etc.

        global_step: int.

        input_words: ...

        ...

    """

    def __init__(self, config, outpath):
        """
        Arguments:

            config: object. Configuration object containing n_epochs,
                n_layers, etc.

            outpath: str. path to where graph should be saved.
        """
        self.outpath = outpath
        self.config = config
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.inputs = None
        self.labels = None
        self.seqlens = None
        self.input_words = None
        self.input_labels = None
        self.input_seqlens = None
        self.embed_matrix = None
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.seqlens_placeholder = None
        self.dropout_placeholder = None
        self.embed_placeholder = None
        self.writer = None
        self.writer_eval = None
        self.pretrained_embeddings = None
        self.summary_op = None
        self.summary_train_op = None
        self.summary_eval_op = None
        self.pred = None
        self.loss = None
        self.train_op = None
        self.accuracy_train = None
        self.f1_train = None
        self.recall_train = None
        self.precision_train = None
        self.accuracy_eval = None
        self.f1_eval = None
        self.recall_eval = None
        self.precision_eval = None

    def build(self):
        """builds computational graph."""
        if self.config.verbosity > 0:
            print('Initializing graph...')
        self._add_placeholders()
        self._init_evaluation_metrics()
        self._init_embedding()
        self.pred = self._add_prediction_op()
        self.loss = self._add_loss_op(self.pred)
        self.train_op = self._add_training_op(self.loss)
        self._create_summaries()
        self._create_evaluation_summaries()

    def train(self,
              X: np.array,
              y: np.array,
              sess: tf.Session,
              saver: tf.train.Saver,
              writer: tf.summary.FileWriter,
              seqlens: np.array = None,
              pretrained_embeddings = None,
              evaluate_kws: dict = None,
              ) -> None:
        """trains tensorflow classifier on all epochs.

        Arguments:

            X: np.array with shape (n_examples, ). Array of inputs.

            y: np.array with shape (n_examples, ). An array of labels to be
                predicted.

            sess: tf.Session().

            saver: tf.train.Saver()

            writer: tf.summary.FileWriter().

            seqlens: np.array with shape (n_example, ). Array of sequence lengths.
                Used in RNN models.

            pretrained_embeddings: np.ndarray with shape (n_examples, embed_size).
                Pretrained embeddings.

            evaluate_kws: dict. Dict of keyword arguments to pass to self.evaluate.

        Returns:

            None.
        """
        # time0 = time.time()
        if self.writer is None:
            self.writer = writer
        # KLUDGE: this is an awkward implementation with evaluate_kws['writer']
        if evaluate_kws is not None and 'writer' in evaluate_kws:
            self.writer_eval = evaluate_kws.pop('writer')
        self.pretrained_embeddings = pretrained_embeddings
        if self.config.debug:
            print('\nTRAINING IN DEBUG MODE.\n')
        if self.config.verbosity > 0:
            print('Training classifier with {0} training examples ({1} unique inputs, {2} unique labels)'.format(
                len(X), self.config.n_inputs, self.config.n_classes)
            )
        sess.run(self.input_words.initializer, feed_dict={self.inputs_placeholder: X})
        sess.run(self.input_labels.initializer, feed_dict={self.labels_placeholder: y})
        if str(self) in ['rnn', 'gru', 'lstm']:
            assert seqlens is not None, 'seqlens must be provided for {0} classifier'.format(self)
            seqlens = np.clip(seqlens, 0, self.config.max_seqlen)
            sess.run(self.input_seqlens.initializer, feed_dict={self.seqlens_placeholder: seqlens})
        if pretrained_embeddings is not None:
            sess.run(self.embed_matrix.initializer, feed_dict={self.embed_placeholder: pretrained_embeddings})
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            n_batches = len(X) / self.config.batch_size
            prog = Progbar(target=1 + n_batches, verbosity=self.config.verbosity)
            evaluate = False
            step = 0
            epoch = 0
            epoch_step = 0
            start_time = time.time()
            while not coord.should_stop():
                # Runs one step of the model.
                if (step + 1) % self.config.eval_every == 0:
                    evaluate = True
                loss, step, preds, labels = self._train_batch(sess, predict=evaluate)
                # if step % self.config.eval_every == 0:
                #     self.evaluate_on_batch(sess, train=True)
                epoch_step += 1
                prog.update(epoch_step, [("train loss", loss)])
                if evaluate:  # evaluates performance.
                    self._evaluate_batch(sess, preds, labels, eval_set=False)  # evaluate on training set.
                    if evaluate_kws is not None:
                        loss_eval, preds_eval, labels_eval = self._predict_batch(sess=sess, **evaluate_kws)  # evaluate on evaluation set.
                        self._evaluate_batch(sess, preds_eval, labels_eval, eval_set=True)  # evaluate on evaluation set.
                    evaluate = False
                if (step + 1) % self.config.save_every == 0:  # saves every n steps.
                    saver.save(sess, os.path.join(self.outpath, 'step'), global_step=self.global_step)
                if epoch_step >= n_batches:
                    duration = (time.time() - start_time) / 60.0
                    epoch += 1
                    epoch_step = 0
                    print('\tFinished epoch {0} of {1}. Total duration: {2:.2f} minutes'.format(epoch, self.config.n_epochs, duration))
                    prog = Progbar(target=1 + n_batches, verbosity=self.config.verbosity)
        except tf.errors.OutOfRangeError:
            if preds is not None:
                self._evaluate_batch(sess, preds, labels, eval_set=False)
            if evaluate_kws is not None:
                loss_eval, preds_eval, labels_eval = self._predict_batch(sess=sess, **evaluate_kws)  # evaluate on evaluation set.
                self._evaluate_batch(sess, preds_eval, labels_eval, eval_set=True)  # evaluate on evaluation set.
            print('\nSaving...')
            saver.save(sess, os.path.join(self.outpath, 'step'), global_step=self.global_step)
            print('Done training for {0} epochs, {1} steps.'.format(self.config.n_epochs, step))
        finally:
        # When done, ask the threads to stop.
            coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
        # sess.close()
        return None

    def evaluate(self, X, y, sess, writer, seqlens=None, pretrained_embeddings=None):
        """evaluates classifier performance.
        """
        if self.writer_eval is None:
            self.writer_eval = writer
        if self.config.debug:
            print('\nEVALUATING IN DEBUG MODE.\n')
        if self.config.verbosity:
            print('Evaluating classifier with {0} observations ({1} unique inputs, {2} unique labels)'.format(
                len(X), self.config.n_inputs, self.config.n_classes)
            )
        sess.run(self.input_words.initializer, feed_dict={self.inputs_placeholder: X})
        sess.run(self.input_labels.initializer, feed_dict={self.labels_placeholder: y})
        if str(self) in ['rnn', 'gru', 'lstm']:
            assert seqlens is not None, 'seqlens must be provided for {0} classifier'.format(self)
            seqlens = np.clip(seqlens, 0, self.config.max_seqlen)
            sess.run(self.input_seqlens.initializer, feed_dict={self.seqlens_placeholder: seqlens})
        if pretrained_embeddings is not None:
            sess.run(self.embed_matrix.initializer, feed_dict={self.embed_placeholder: pretrained_embeddings})
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            n_batches = len(X) / self.config.batch_size
            prog = Progbar(target=1 + n_batches, verbosity=self.config.verbosity)
            step = 0
            epoch = 0
            epoch_step = 0
            start_time = time.time()
            while not coord.should_stop():
                loss, preds, labels = self._predict_batch(sess)
                perf = self._evaluate_batch(sess, preds, labels, eval_set=True)
                # if step % self.config.eval_every == 0:
                epoch_step += 1
                prog.update(epoch_step, [("evaluation loss", loss)])
                if epoch_step >= n_batches:
                    duration = (time.time() - start_time) / 60.0
                    epoch += 1
                    epoch_step = 0
                    print('\tFinished epoch {0} of {1}. Total duration: {2:.2f} minutes'.format(epoch, self.config.n_epochs, duration))
                    prog = Progbar(target=1 + n_batches, verbosity=self.config.verbosity)
        except tf.errors.OutOfRangeError:
            # print('Saving...')
            # saver.save(sess, os.path.join(self.config.path, self.name, self.__str__()), global_step=self.global_step)
            print('Done evaluating for {0} epochs, {1} steps.'.format(self.config.n_epochs, step))
            print(' f1 score: {0:3f}'.format(perf['f1']))
            print(' precision: {0:3f}'.format(perf['precision']))
            print(' recall: {0:3f}'.format(perf['recall']))
            print(' accuracy: {0:3f}'.format(perf['accuracy']))
            print(' confusion matrix:\n{0}'.format(perf['confusion']))
            heatmap_confusion(perf['confusion'], os.path.join(self.outpath, 'confusion_matrix_eval.png'))
        finally:
        # When done, ask the threads to stop.
            coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
        # sess.close()

    def restore(self, sess, saver):
        """attempts to restore existing model within self.outpath.

        Creats self.outpath directory if it does not exist.

        Returns:

            None.
        """
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(self.outpath, 'checkpoint')))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            if self.config.verbosity > 0:
                print('Restored classifier from {0}.'.format(self.outpath))
        else:
            if self.config.verbosity > 0:
                print('No classifier to restore in {0}.'.format(self.outpath))

    def _add_placeholders(self):
        raise NotImplementedError('Must be implemented by child class.')

    def _init_embedding(self):
        """Initializes input embedding matrix.

        Shape of embedding matrix: n_inputs X embed_size.
        """
        # with tf.device('/cpu:0'):
        with tf.name_scope("embed"):
            if self.config.use_pretrained:
                # self.embed_matrix = tf.placeholder(tf.float32, shape=(self.config.n_inputs, self.config.embed_size), name='embed_matrix')
                # embed_matrix = tf.Variable(tf.constant(0.0, shape=[self.config.n_inputs, self.config.embed_size]), trainable=False, name='embed_matrix')
                self.embed_placeholder = tf.placeholder(
                    tf.float32,
                    shape=[self.config.n_inputs, self.config.embed_size],
                    name='embed_matrix'
                )
                self.embed_matrix = tf.Variable(self.embed_placeholder, trainable=True, collections=[])
                # self.embed_matrix = embed_matrix.assign(self.embed_placeholder)
            else:
                self.embed_matrix = tf.Variable(
                    tf.random_uniform([self.config.n_inputs, self.config.embed_size], -0.1, 0.1),
                    name='embed_matrix'
                )

    def _add_prediction_op(self):
        raise NotImplementedError('Must be implemented by child class.')

    def _add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.

        Arguments:

            pred: A tensor of shape (batch_size, n_classes).

        Returns:

            loss: A 0-d tensor (scalar) output.
        """
        # with tf.device('/cpu:0'):
        with tf.name_scope("loss"):
            # y_reshaped = tf.reshape(self.labels, [-1])
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=self.labels)
            loss = tf.reduce_mean(entropy)
        return loss

    def _add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Arguments:

            loss: Loss tensor (a scalar).

        Returns:

            train_op: The Op for training.
        """
        # with tf.device('/cpu:0'):
        # train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(loss, global_step=self.global_step)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        grads, params = zip(*optimizer.compute_gradients(loss))
        if self.config.clip_gradients:
            grads, _ = tf.clip_by_global_norm(grads, self.config.max_grad_norm)
        # self.grad_norm = tf.global_norm(grads)
        train_op = optimizer.apply_gradients(zip(grads, params), global_step=self.global_step)
        return train_op

    def _train_batch(self, sess, predict=False):
        """Perform one step of gradient descent on the provided batch of data.

        Arguments:

            sess: tf.Session().

            predict: bool = False. If True, computes predicted values.

        Returns:

            loss: loss over the batch (a scalar).
        """
        # feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        feed_dict = {}
        if self.dropout_placeholder is not None:
            feed_dict = {self.dropout_placeholder: self.config.p_keep}
        if predict:
            inputs, labels, _, loss, summary, step, preds = sess.run(
                [self.inputs, self.labels, self.train_op, self.loss, self.summary_op, self.global_step, self.pred],
                feed_dict=feed_dict
            )
            result = (loss, step, preds, labels)
        else:
            inputs, labels, _, loss, summary, step = sess.run(
                [self.inputs, self.labels, self.train_op, self.loss, self.summary_op, self.global_step],
                feed_dict=feed_dict
            )
            result = (loss, step, None, None)
        self.writer.add_summary(summary, step)
        return result

    def _predict_batch(self, sess, inputs=None, labels=None, seqlens=None):
        """

        Arguments:

            sess: tf.Session().

            inputs: np.array (default: None).

            labels: np.array (default: None).

            seqlens: np.array (default: None).

        Returns:

            loss, preds, labels: tuple.

                loss: np.float64. Prediction loss.

                preds: np.array. Array of predicted values.

                labels: np.array. Array of labels.
        """
        assert (inputs is None and labels is None) or (inputs is not None and labels is not None), 'either inputs and labels must both be None or both be not None.'
        feed_dict = {}
        if self.dropout_placeholder is not None:
            feed_dict = {self.dropout_placeholder: 1.0}
        if inputs is None and labels is None:
            loss, summary, step, preds, inputs, labels = sess.run(
                [self.loss, self.summary_op, self.global_step, self.pred, self.inputs, self.labels],
                feed_dict=feed_dict
            )
        else:
            feed_dict.update({
                self.inputs: inputs,
                self.labels: labels,
            })
            if seqlens is not None:
                feed_dict[self.seqlens] = seqlens
            loss, summary, step, preds = sess.run(
                [self.loss, self.summary_op, self.global_step, self.pred],
                feed_dict=feed_dict
            )
        return loss, preds, labels

    def _evaluate_batch(self, sess, preds, labels, eval_set=False):
        """
        Returns:

            perf: dict. Dict containing model performance.
        """
        try:
            feed_dict = {}
            if self.dropout_placeholder is not None:
                feed_dict = {self.dropout_placeholder: self.config.p_keep}
            preds_class = np.argmax(preds, 1)
            # preds_class = sess.run(preds_class)
            perf = self._update_evaluation_metrics(sess, preds_class, labels)
            # eval_summary_op = self.create_evaluation_summaries()
            if eval_set:
                summary, step = sess.run([self.summary_eval_op, self.global_step], feed_dict=feed_dict)
                self.writer_eval.add_summary(summary, step)
            else:
                summary, step = sess.run([self.summary_train_op, self.global_step], feed_dict=feed_dict)
                self.writer.add_summary(summary, step)
            if self.config.verbosity > 0:
                batch = 'eval' if eval_set else 'train'
                print('\nf1 score on {0} set: {1:3f}'.format(batch, perf['f1']))
                print('precision on {0} set: {1:3f}'.format(batch, perf['precision']))
                print('recall on {0} set: {1:3f}'.format(batch, perf['recall']))
                print('accuracy on {0} set: {1:3f}'.format(batch, perf['accuracy']))
                print('confusion matrix on {0} set:\n{1}'.format(batch, perf['confusion']))
                heatmap_confusion(perf['confusion'], os.path.join(self.outpath, 'confusion_matrix_{0}.png'.format(batch)))
        except ValueError as err:
            print('\nEncountered error in batch evaluation.')
            print(err)
            traceback.print_exc()
            perf = {
                'accuracy': 0.0,
                'recall': 0.0,
                'precision': 0.0,
                'f1': 0.0,
                'confusion': np.nan,
                # 'roc': roc
            }
        return perf

    def _update_evaluation_metrics(self, sess, preds_batch, labels_batch, eval_set=False):
        """Evaluates predictions given a sample of inputs and labels.

        Arguments:

            sess: tf.Session().

            preds_batch: A batch of label predictions.

            labels_batch: A batch of label data.

        Returns:

            perf: dict. Dict containing model performance.
        """
        # print(sess.run(tf.argmax(preds, 1)))
        # pred_class = tf.argmax(preds, 1)
        # correct_prediction = tf.equal(pred_class, labels_batch)
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # pred_class, acc = sess.run(
        #   [pred_class, accuracy],
        #   {self.inputs_placeholder: inputs_batch, self.labels_placeholder: labels_batch}
        # )
        accuracy = metrics.accuracy_score(labels_batch, preds_batch)
        recall = metrics.recall_score(labels_batch, preds_batch, average='macro')
        precision = metrics.precision_score(labels_batch, preds_batch, average='macro')
        f1_macro = metrics.f1_score(labels_batch, preds_batch, average='macro')
        confusion = metrics.confusion_matrix(labels_batch, preds_batch)
        if eval_set:
            sess.run([
                self.accuracy_eval.assign(accuracy),
                self.f1_eval.assign(f1_macro),
                self.recall_eval.assign(recall),
                self.precision_eval.assign(precision),
            ])
        else:
            sess.run([
                self.accuracy_train.assign(accuracy),
                self.f1_train.assign(f1_macro),
                self.recall_train.assign(recall),
                self.precision_train.assign(precision),
            ])
        # roc = metrics.roc_curve(labels_batch, preds)
        perf = {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1': f1_macro,
            'confusion': confusion,
            # 'roc': roc
        }
        return perf

    def _init_evaluation_metrics(self):
        with tf.name_scope('summary_train'):
            self.accuracy_train = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='accuracy_train')
            self.f1_train = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='f1_train')
            self.recall_train = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='recall_train')
            self.precision_train = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='precision_train')
        with tf.name_scope('summary_eval'):
            self.accuracy_eval = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='accuracy_eval')
            self.f1_eval = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='f1_eval')
            self.recall_eval = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='recall_eval')
            self.precision_eval = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='precision_eval')

    def _create_summaries(self):
        """Creates summary plots for tensorboard."""
        with tf.name_scope("loss_summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram_loss", self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def _create_evaluation_summaries(self):
        """Creates model evaluation summary plots for tensorboard."""
        with tf.name_scope("train_score_summaries"):
            acc = tf.summary.scalar("accuracy", self.accuracy_train)
            f1 = tf.summary.scalar("f1", self.f1_train)
            prec = tf.summary.scalar("precision", self.precision_train)
            rec = tf.summary.scalar("recall", self.recall_train)
            # tf.summary.histogram("histogram_loss", self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_train_op = tf.summary.merge([acc, f1, prec, rec])
        with tf.name_scope("eval_score_summaries"):
            acc = tf.summary.scalar("accuracy", self.accuracy_eval)
            f1 = tf.summary.scalar("f1", self.f1_eval)
            prec = tf.summary.scalar("precision", self.precision_eval)
            rec = tf.summary.scalar("recall", self.recall_eval)
            # tf.summary.histogram("histogram_loss", self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_eval_op = tf.summary.merge([acc, f1, prec, rec])
