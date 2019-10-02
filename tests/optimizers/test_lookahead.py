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
        eager = False # seems to help with coverage

        tf.compat.v1.reset_default_graph()
        if eager:
            tf.compat.v1.enable_eager_execution()
        else:
            tf.compat.v1.disable_eager_execution()
        tf.compat.v1.set_random_seed(7411)

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
            for use_lookahead in [True, False, False, True]:
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

'''
(opt:    Adam, LA:  True impl: callback) trained in 285.4 steps (std:  17.6): [298, 295, 297, 300, 261, 284, 251, 268, 300, 300] final loss: 0.196 (0.037) [0.116, 0.121, 0.079, 0.270, 0.136, 0.122, 0.140, 0.107, 0.493, 0.377] acc: 0.9402 val_acc: [0.961, 0.963, 0.983, 0.906, 0.961, 0.960, 0.957, 0.963, 0.855, 0.893]
(opt:    Adam, LA: False impl: callback) trained in 300.0 steps (std:   0.0): [300, 300, 300, 300, 300, 300, 300, 300, 300, 300] final loss: 0.666 (0.120) [0.299, 0.228, 0.418, 0.217, 0.519, 0.456, 1.075, 2.626, 0.243, 0.582] acc: 0.8025 val_acc: [0.884, 0.923, 0.838, 0.941, 0.805, 0.845, 0.664, 0.407, 0.907, 0.811]
(opt:    Adam, LA: False impl: callback) trained in 300.0 steps (std:   0.0): [300, 300, 300, 300, 300, 300, 300, 300, 300, 300] final loss: 0.576 (0.156) [0.284, 1.082, 0.513, 0.901, 0.563, 0.765, 0.462, 0.342, 0.278, 0.570] acc: 0.7991 val_acc: [0.893, 0.658, 0.808, 0.693, 0.793, 0.729, 0.833, 0.867, 0.906, 0.811]
(opt:    Adam, LA:  True impl: callback) trained in 264.4 steps (std:  22.5): [300, 271, 223, 255, 265, 235, 265, 297, 268, 265] final loss: 0.271 (0.019) [1.516, 0.137, 0.154, 0.145, 0.103, 0.162, 0.144, 0.094, 0.116, 0.141] acc: 0.9343 val_acc: [0.668, 0.962, 0.967, 0.965, 0.971, 0.954, 0.954, 0.978, 0.965, 0.959]
(opt:   RAdam, LA:  True impl: callback) trained in 300.0 steps (std:   0.0): [300, 300, 300, 300, 300, 300, 300, 300, 300, 251] final loss: 0.263 (0.045) [0.254, 0.271, 0.223, 0.361, 0.242, 0.251, 0.217, 0.485, 0.174, 0.149] acc: 0.9074 val_acc: [0.909, 0.903, 0.917, 0.858, 0.933, 0.921, 0.908, 0.816, 0.948, 0.961]
(opt:   RAdam, LA: False impl: callback) trained in 300.0 steps (std:   0.0): [300, 300, 300, 300, 300, 300, 300, 300, 300, 300] final loss: 1.087 (0.428) [1.549, 0.764, 0.712, 2.246, 0.752, 0.525, 0.659, 0.850, 1.360, 1.458] acc: 0.7094 val_acc: [0.643, 0.761, 0.805, 0.507, 0.737, 0.827, 0.800, 0.764, 0.589, 0.660]
(opt:   RAdam, LA: False impl: callback) trained in 300.0 steps (std:   0.0): [300, 300, 300, 300, 300, 300, 300, 300, 300, 300] final loss: 0.830 (0.197) [0.616, 0.570, 1.159, 0.592, 1.015, 0.646, 0.776, 0.726, 0.455, 1.751] acc: 0.7383 val_acc: [0.776, 0.833, 0.662, 0.801, 0.662, 0.762, 0.803, 0.740, 0.840, 0.504]
(opt:   RAdam, LA:  True impl: callback) trained in 294.8 steps (std:   7.4): [300, 300, 286, 300, 300, 300, 283, 300, 284, 267] final loss: 0.273 (0.029) [0.313, 0.284, 0.136, 0.384, 0.320, 0.618, 0.142, 0.217, 0.166, 0.149] acc: 0.9132 val_acc: [0.915, 0.905, 0.959, 0.889, 0.874, 0.780, 0.962, 0.938, 0.954, 0.958]
(opt: RMSprop, LA:  True impl: callback) trained in 300.0 steps (std:   0.0): [300, 300, 300, 300, 300, 300, 300, 300, 300, 300] final loss: 0.367 (0.076) [0.329, 0.521, 0.239, 0.395, 0.403, 0.356, 0.297, 0.612, 0.236, 0.280] acc: 0.8723 val_acc: [0.893, 0.776, 0.926, 0.880, 0.858, 0.899, 0.915, 0.736, 0.923, 0.917]
(opt: RMSprop, LA: False impl: callback) trained in 300.0 steps (std:   0.0): [300, 300, 300, 300, 300, 300, 294, 300, 300, 300] final loss: 0.755 (0.410) [0.740, 1.809, 0.915, 0.414, 0.521, 0.869, 0.157, 0.735, 0.785, 0.610] acc: 0.8340 val_acc: [0.859, 0.692, 0.762, 0.870, 0.840, 0.765, 0.968, 0.819, 0.871, 0.892]
(opt: RMSprop, LA: False impl: callback) trained in 300.0 steps (std:   0.0): [300, 300, 300, 300, 300, 300, 300, 293, 300, 293] final loss: 0.532 (0.233) [0.366, 0.670, 0.471, 0.510, 0.564, 0.559, 1.038, 0.133, 0.845, 0.165] acc: 0.8765 val_acc: [0.921, 0.823, 0.854, 0.888, 0.884, 0.844, 0.770, 0.965, 0.859, 0.956]
(opt: RMSprop, LA:  True impl: callback) trained in 300.0 steps (std:   0.0): [300, 300, 300, 300, 300, 300, 300, 300, 300, 300] final loss: 0.613 (0.230) [0.578, 0.550, 0.213, 0.446, 0.292, 0.448, 0.680, 1.418, 0.393, 1.110] acc: 0.8094 val_acc: [0.771, 0.807, 0.926, 0.860, 0.914, 0.875, 0.760, 0.648, 0.905, 0.627]
(opt:     SGD, LA:  True impl: callback) trained in 300.0 steps (std:   0.0): [300, 300, 300, 300, 300, 300, 300, 300, 300, 300] final loss: 1.800 (0.000) [1.744, 1.801, 1.814, 1.809, 1.812, 1.824, 1.788, 1.794, 1.782, 1.829] acc: 0.2810 val_acc: [0.275, 0.281, 0.278, 0.287, 0.272, 0.271, 0.312, 0.277, 0.275, 0.282]
(opt:     SGD, LA: False impl: callback) trained in 300.0 steps (std:   0.0): [300, 300, 300, 300, 300, 300, 300, 300, 300, 300] final loss: 1.563 (0.005) [1.511, 1.575, 1.555, 1.552, 1.454, 1.339, 1.757, 1.600, 1.673, 1.617] acc: 0.4001 val_acc: [0.400, 0.373, 0.469, 0.394, 0.449, 0.496, 0.314, 0.399, 0.351, 0.356]
(opt:     SGD, LA: False impl: callback) trained in 300.0 steps (std:   0.0): [300, 300, 300, 300, 300, 300, 300, 300, 300, 300] final loss: 1.587 (0.006) [1.730, 1.734, 1.553, 1.637, 1.573, 1.588, 1.488, 1.542, 1.587, 1.437] acc: 0.3825 val_acc: [0.329, 0.301, 0.378, 0.391, 0.406, 0.383, 0.423, 0.414, 0.376, 0.424]
(opt:     SGD, LA:  True impl: callback) trained in 300.0 steps (std:   0.0): [300, 300, 300, 300, 300, 300, 300, 300, 300, 300] final loss: 1.803 (0.000) [1.807, 1.830, 1.799, 1.786, 1.770, 1.777, 1.839, 1.762, 1.837, 1.818] acc: 0.2833 val_acc: [0.242, 0.265, 0.285, 0.287, 0.314, 0.326, 0.266, 0.286, 0.250, 0.313]


(opt:    Adam, LA:  True impl: callback) final loss: 0.196 (0.037) [0.116, 0.121, 0.079, 0.270, 0.136, 0.122, 0.140, 0.107, 0.493, 0.377] acc: 0.9402 val_acc: [0.961, 0.963, 0.983, 0.906, 0.961, 0.960, 0.957, 0.963, 0.855, 0.893]
(opt:    Adam, LA: False impl: callback) final loss: 0.666 (0.120) [0.299, 0.228, 0.418, 0.217, 0.519, 0.456, 1.075, 2.626, 0.243, 0.582] acc: 0.8025 val_acc: [0.884, 0.923, 0.838, 0.941, 0.805, 0.845, 0.664, 0.407, 0.907, 0.811]
(opt:    Adam, LA: False impl: callback) final loss: 0.576 (0.156) [0.284, 1.082, 0.513, 0.901, 0.563, 0.765, 0.462, 0.342, 0.278, 0.570] acc: 0.7991 val_acc: [0.893, 0.658, 0.808, 0.693, 0.793, 0.729, 0.833, 0.867, 0.906, 0.811]
(opt:    Adam, LA:  True impl: callback) final loss: 0.271 (0.019) [1.516, 0.137, 0.154, 0.145, 0.103, 0.162, 0.144, 0.094, 0.116, 0.141] acc: 0.9343 val_acc: [0.668, 0.962, 0.967, 0.965, 0.971, 0.954, 0.954, 0.978, 0.965, 0.959]
(opt:   RAdam, LA:  True impl: callback) final loss: 0.263 (0.045) [0.254, 0.271, 0.223, 0.361, 0.242, 0.251, 0.217, 0.485, 0.174, 0.149] acc: 0.9074 val_acc: [0.909, 0.903, 0.917, 0.858, 0.933, 0.921, 0.908, 0.816, 0.948, 0.961]
(opt:   RAdam, LA: False impl: callback) final loss: 1.087 (0.428) [1.549, 0.764, 0.712, 2.246, 0.752, 0.525, 0.659, 0.850, 1.360, 1.458] acc: 0.7094 val_acc: [0.643, 0.761, 0.805, 0.507, 0.737, 0.827, 0.800, 0.764, 0.589, 0.660]
(opt:   RAdam, LA: False impl: callback) final loss: 0.830 (0.197) [0.616, 0.570, 1.159, 0.592, 1.015, 0.646, 0.776, 0.726, 0.455, 1.751] acc: 0.7383 val_acc: [0.776, 0.833, 0.662, 0.801, 0.662, 0.762, 0.803, 0.740, 0.840, 0.504]
(opt:   RAdam, LA:  True impl: callback) final loss: 0.273 (0.029) [0.313, 0.284, 0.136, 0.384, 0.320, 0.618, 0.142, 0.217, 0.166, 0.149] acc: 0.9132 val_acc: [0.915, 0.905, 0.959, 0.889, 0.874, 0.780, 0.962, 0.938, 0.954, 0.958]
(opt: RMSprop, LA:  True impl: callback) final loss: 0.367 (0.076) [0.329, 0.521, 0.239, 0.395, 0.403, 0.356, 0.297, 0.612, 0.236, 0.280] acc: 0.8723 val_acc: [0.893, 0.776, 0.926, 0.880, 0.858, 0.899, 0.915, 0.736, 0.923, 0.917]
(opt: RMSprop, LA: False impl: callback) final loss: 0.755 (0.410) [0.740, 1.809, 0.915, 0.414, 0.521, 0.869, 0.157, 0.735, 0.785, 0.610] acc: 0.8340 val_acc: [0.859, 0.692, 0.762, 0.870, 0.840, 0.765, 0.968, 0.819, 0.871, 0.892]
(opt: RMSprop, LA: False impl: callback) final loss: 0.532 (0.233) [0.366, 0.670, 0.471, 0.510, 0.564, 0.559, 1.038, 0.133, 0.845, 0.165] acc: 0.8765 val_acc: [0.921, 0.823, 0.854, 0.888, 0.884, 0.844, 0.770, 0.965, 0.859, 0.956]
(opt: RMSprop, LA:  True impl: callback) final loss: 0.613 (0.230) [0.578, 0.550, 0.213, 0.446, 0.292, 0.448, 0.680, 1.418, 0.393, 1.110] acc: 0.8094 val_acc: [0.771, 0.807, 0.926, 0.860, 0.914, 0.875, 0.760, 0.648, 0.905, 0.627]
(opt:     SGD, LA:  True impl: callback) final loss: 1.800 (0.000) [1.744, 1.801, 1.814, 1.809, 1.812, 1.824, 1.788, 1.794, 1.782, 1.829] acc: 0.2810 val_acc: [0.275, 0.281, 0.278, 0.287, 0.272, 0.271, 0.312, 0.277, 0.275, 0.282]
(opt:     SGD, LA: False impl: callback) final loss: 1.563 (0.005) [1.511, 1.575, 1.555, 1.552, 1.454, 1.339, 1.757, 1.600, 1.673, 1.617] acc: 0.4001 val_acc: [0.400, 0.373, 0.469, 0.394, 0.449, 0.496, 0.314, 0.399, 0.351, 0.356]
(opt:     SGD, LA: False impl: callback) final loss: 1.587 (0.006) [1.730, 1.734, 1.553, 1.637, 1.573, 1.588, 1.488, 1.542, 1.587, 1.437] acc: 0.3825 val_acc: [0.329, 0.301, 0.378, 0.391, 0.406, 0.383, 0.423, 0.414, 0.376, 0.424]
(opt:     SGD, LA:  True impl: callback) final loss: 1.803 (0.000) [1.807, 1.830, 1.799, 1.786, 1.770, 1.777, 1.839, 1.762, 1.837, 1.818] acc: 0.2833 val_acc: [0.242, 0.265, 0.285, 0.287, 0.314, 0.326, 0.266, 0.286, 0.250, 0.313]

(opt:    Adam, LA:  True impl: callback) acc: 0.9402 val_acc: [0.961, 0.963, 0.983, 0.906, 0.961, 0.960, 0.957, 0.963, 0.855, 0.893]
(opt:    Adam, LA: False impl: callback) acc: 0.8025 val_acc: [0.884, 0.923, 0.838, 0.941, 0.805, 0.845, 0.664, 0.407, 0.907, 0.811]
(opt:    Adam, LA: False impl: callback) acc: 0.7991 val_acc: [0.893, 0.658, 0.808, 0.693, 0.793, 0.729, 0.833, 0.867, 0.906, 0.811]
(opt:    Adam, LA:  True impl: callback) acc: 0.9343 val_acc: [0.668, 0.962, 0.967, 0.965, 0.971, 0.954, 0.954, 0.978, 0.965, 0.959]
(opt:   RAdam, LA:  True impl: callback) acc: 0.9074 val_acc: [0.909, 0.903, 0.917, 0.858, 0.933, 0.921, 0.908, 0.816, 0.948, 0.961]
(opt:   RAdam, LA: False impl: callback) acc: 0.7094 val_acc: [0.643, 0.761, 0.805, 0.507, 0.737, 0.827, 0.800, 0.764, 0.589, 0.660]
(opt:   RAdam, LA: False impl: callback) acc: 0.7383 val_acc: [0.776, 0.833, 0.662, 0.801, 0.662, 0.762, 0.803, 0.740, 0.840, 0.504]
(opt:   RAdam, LA:  True impl: callback) acc: 0.9132 val_acc: [0.915, 0.905, 0.959, 0.889, 0.874, 0.780, 0.962, 0.938, 0.954, 0.958]
(opt: RMSprop, LA:  True impl: callback) acc: 0.8723 val_acc: [0.893, 0.776, 0.926, 0.880, 0.858, 0.899, 0.915, 0.736, 0.923, 0.917]
(opt: RMSprop, LA: False impl: callback) acc: 0.8340 val_acc: [0.859, 0.692, 0.762, 0.870, 0.840, 0.765, 0.968, 0.819, 0.871, 0.892]
(opt: RMSprop, LA: False impl: callback) acc: 0.8765 val_acc: [0.921, 0.823, 0.854, 0.888, 0.884, 0.844, 0.770, 0.965, 0.859, 0.956]
(opt: RMSprop, LA:  True impl: callback) acc: 0.8094 val_acc: [0.771, 0.807, 0.926, 0.860, 0.914, 0.875, 0.760, 0.648, 0.905, 0.627]
(opt:     SGD, LA:  True impl: callback) acc: 0.2810 val_acc: [0.275, 0.281, 0.278, 0.287, 0.272, 0.271, 0.312, 0.277, 0.275, 0.282]
(opt:     SGD, LA: False impl: callback) acc: 0.4001 val_acc: [0.400, 0.373, 0.469, 0.394, 0.449, 0.496, 0.314, 0.399, 0.351, 0.356]
(opt:     SGD, LA: False impl: callback) acc: 0.3825 val_acc: [0.329, 0.301, 0.378, 0.391, 0.406, 0.383, 0.423, 0.414, 0.376, 0.424]
(opt:     SGD, LA:  True impl: callback) acc: 0.2833 val_acc: [0.242, 0.265, 0.285, 0.287, 0.314, 0.326, 0.266, 0.286, 0.250, 0.313]


'''