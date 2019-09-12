# coding=utf-8
#
# created by kpe on 05.Sep.2019 at 16:36
#

from __future__ import absolute_import, division, print_function


import unittest
import itertools

import numpy as np
import tensorflow as tf

from tensorflow import keras

import params_flow as pf


class StopTraining(keras.callbacks.Callback):
    """ Stops the training when both train and validation accuracy
    reach the accuracy target (by default 100%). """

    patience = 3
    acc_target = .95  # 0.95

    def __init__(self):
        super(StopTraining, self).__init__()
        self.hit_count = 0

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get("acc")
        if "val_acc" not in logs:
            return
        val_acc = logs.get("val_acc")
        if (self.acc_target - 0.0001) < acc and (self.acc_target - 0.0001) < val_acc:
            self.hit_count += 1
            if self.patience <= self.hit_count:
                self.model.stop_training = True
        else:
            self.hit_count = 0


class LookAheadTest(unittest.TestCase):
    """
    This test case will compare different optimizers with and without lookahead,
    by training a 2-layer FFN/MLP classifier counting the number of ones in a binary sequence.

    N.B. To demonstrate the lookahead stabilizing ability on the training, we'll use
    large learning rates, small batch sizes, and an overparameterized network.
    (as it seems - when the gradient updates are less noisy, lookahead might not have a
    noticeable effect and might event inhibit the training).
    """
    seq_len    = 10
    batch_size = 8
    ds_size    = int(5e6)

    def setUp(self) -> None:
        eager = False

        tf.compat.v1.reset_default_graph()
        if eager:
            tf.compat.v1.enable_eager_execution()
        else:
            tf.compat.v1.disable_eager_execution()
        tf.set_random_seed(7411)

        self.train_ds = self.get_ds()
        self.valid_ds = self.get_ds(shuffle=False).take(2**self.seq_len)

    def tearDown(self) -> None:
        del self.train_ds
        del self.valid_ds
        keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

    @staticmethod
    def get_ds(seq_len=seq_len, batch_size=batch_size, shuffle=True):
        """ Returns a dataset of binary sequences labeled with the number of ones in the sequence. """

        # all possible binary permutations of len seq_len
        perms = [perm for perm in itertools.product([0, 1], repeat=seq_len)]

        def to_train_pair(x):
            y = tf.reduce_sum(x, axis=-1)    # ones count
            return tf.cast(x, tf.float32), y

        # dataset from random tensors of zeros and ones
        ds = tf.data.Dataset.from_tensors(perms).repeat()
        if shuffle:
            ds = ds.shuffle(100000, reshuffle_each_iteration=True)
        ds = ds.map(to_train_pair)
        ds = ds.apply(tf.data.experimental.unbatch())
        ds = ds.batch(batch_size)
        return ds

    def test_cover_callback(self):
        model = self.build_model(pf.optimizers.RAdam(), False)
        lookahead_callback = pf.optimizers.LookaheadOptimizerCallback()
        model.fit(self.get_ds(), steps_per_epoch=2, epochs=20, callbacks=[lookahead_callback])

    def test_cover_wrapper(self):
        model = self.build_model(pf.optimizers.RAdam(), True)
        model.fit(self.get_ds(), steps_per_epoch=2, epochs=20)

    def test_lookahead_callback(self):
        self.compare_optimizers_test_lookahead(use_wrapper=False)

    def test_lookahead_wrapper(self):
        self.compare_optimizers_test_lookahead(use_wrapper=True)

    def build_model(self, optimizer, use_wrapper):
        model = keras.Sequential([
            keras.layers.Dense(self.seq_len * 4, activation="relu"),
            keras.layers.Dense(self.seq_len * 4, activation="relu"),
            keras.layers.Dense(self.seq_len * 4, activation="relu"),
            keras.layers.Dense(self.seq_len + 1, activation="softmax"),
        ])
        model.compile(optimizer=optimizer,
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])
        model.build((self.batch_size, self.seq_len))

        # regularization
        #pf.utils.add_dense_layer_loss(model,
        #                              kernel_regularizer=keras.regularizers.l1_l2(1e-3, 1e-4),
        #                              bias_regularizer=keras.regularizers.l1_l2(1e-3, 1e-4))

        if use_wrapper:
            lookahead = pf.optimizers.lookahead.OptimizerLookaheadWrapper()
            model = lookahead.wrap(model)

        return model

    def train_model(self, optimizer, use_lookahead, use_wrapper):
        model = self.build_model(optimizer, use_wrapper)

        callbacks = []
        if use_lookahead:
            if not use_wrapper:
                callbacks = [pf.optimizers.LookaheadOptimizerCallback()]

        hist = model.fit(self.train_ds,
                         steps_per_epoch=8,
                         epochs=300,
                         validation_data=self.valid_ds,
                         validation_steps=2**self.seq_len/self.batch_size,
                         validation_freq=1,
                         callbacks=callbacks + [StopTraining()],
                         verbose=0)

        epoch_acc  = hist.history['val_acc']
        final_loss = hist.history['val_loss'][-3:]
        return len(epoch_acc), np.mean(final_loss), np.std(final_loss), np.mean(epoch_acc[-3:])

    @staticmethod
    def reject_outliers(data, m=2):
        res = data
        outliers = abs(data - np.mean(data)) < m * np.std(data)
        if any(outliers):
            res = data[outliers]
        return res

    def compare_optimizers_test_lookahead(self, use_wrapper=True):
        return  # don't run in CI as this takes time

        lr = 5e-3           # using large learning rate to provoke unstable training
        repeat_count = 10

        print("learning_rate", lr)
        print(" repeat_count", repeat_count)
        print("        eager", tf.executing_eagerly())
        print("         impl", "wrapper" if use_wrapper else "callback")

        for optimizer_type in [keras.optimizers.Adam, pf.optimizers.RAdam,
                               keras.optimizers.RMSprop, keras.optimizers.SGD]:
            for use_lookahead in [True, False, True, False]:
                hist   = [self.train_model(optimizer_type(learning_rate=lr), use_lookahead, use_wrapper)
                          for _ in range(repeat_count)]

                epochs, loss, loss_std, acc = map(np.array, zip(*hist))
                clean_epochs = self.reject_outliers(epochs)

                print("(opt:{:>8s}, LA:{:>6s} impl:{:>9s}) trained in {:5.1f} steps (std: {:5.1f}): [{}]".format(
                    optimizer_type.__name__, str(use_lookahead),
                    "wrapper" if use_wrapper else "callback",
                    clean_epochs.mean(), clean_epochs.std(),
                    ", ".join([("{}" if abs(steps - clean_epochs.mean()) <= 2*clean_epochs.std() else "({})").format(steps)
                               for steps in epochs])),
                    "final loss: {:.3f} ({:.3f})".format(loss.mean(), loss_std.mean()),
                    "[{}]".format(", ".join(["{:.3f}".format(l) for l in loss])),
                    "acc: {:.4f}".format(acc.mean()),
                    "val_acc: [{}]".format(", ".join(["{:.3f}".format(a) for a in acc]))
                )
