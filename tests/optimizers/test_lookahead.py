# coding=utf-8
#
# created by kpe on 05.Sep.2019 at 16:36
#

from __future__ import absolute_import, division, print_function


import unittest

import numpy as np
import tensorflow as tf

from tensorflow import keras

import params_flow as pf


class LookAheadTest(unittest.TestCase):
    seq_len    = 4
    batch_size = 64
    ds_size    = int(5e6)

    def setUp(self) -> None:
        eager = False

        tf.compat.v1.reset_default_graph()
        if eager:
            tf.compat.v1.enable_eager_execution()
        else:
            tf.compat.v1.disable_eager_execution()

        tf.compat.v1.set_random_seed(4711)
        self.train_ds, self.valid_ds = self.get_ds(), self.get_ds(size=int(1e5))



    @staticmethod
    def get_ds(shape=(seq_len,), size=int(ds_size), batch_size=batch_size):
        # generate in batches (to improve performance)
        ds_shape = [size//10]+list(shape)

        def to_train_pair(x):
            y = tf.reduce_sum(x, axis=-1)    # ones count
            y = tf.math.mod(y, 2)            # ones count mod 2
            return tf.cast(x, tf.float32), y

        # dataset from random tensors of zeros and ones
        ds = tf.data.Dataset.from_tensor_slices(
            [tf.cast(tf.round(tf.random_uniform(ds_shape)), tf.int64) for _ in range(10)])
        ds = ds.shuffle(10000)
        ds = ds.map(to_train_pair)
        ds = ds.apply(tf.data.experimental.unbatch())
        ds = ds.batch(batch_size)
        return ds

    class StopTraining(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            acc = logs.get("acc")
            val_acc = logs.get("val_acc")
            if np.abs(acc - 1.) < 0.0001 and np.abs(val_acc - 1.) < 0.0001:
                self.model.stop_training = True

    def build_model(self, optimizer, use_wrapper):
        model = keras.Sequential([
            keras.layers.Dense(2 ** (self.seq_len+2), activation="tanh"),
            keras.layers.Dense(2, activation="softmax"),
        ])
        model.build((self.batch_size, self.seq_len))
        model.compile(optimizer=optimizer,
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

        if use_wrapper:
            lookahead = pf.optimizers.lookahead.OptimizerLookaheadWrapper()
            model = lookahead.wrap(model)

        return model

    def test_cover_callback(self):
        model = self.build_model(pf.optimizers.RAdam(), False)
        lookahead_callback = pf.optimizers.LookaheadOptimizerCallback()
        model.fit(self.get_ds(), steps_per_epoch=2, epochs=20, callbacks=[lookahead_callback])

    def test_cover_wrapper(self):
        model = self.build_model(pf.optimizers.RAdam(), True)
        model.fit(self.get_ds(), steps_per_epoch=2, epochs=20)

    def train_model(self, optimizer, use_lookahead, use_wrapper):
        model = self.build_model(optimizer, use_wrapper)

        callbacks = [self.StopTraining()]
        if use_lookahead:
            if use_wrapper:
               pass
            else:
                callbacks = [pf.optimizers.LookaheadOptimizerCallback()] + callbacks

        hist = model.fit(self.train_ds,
                         steps_per_epoch=2**(2*self.seq_len),
                         epochs=1000,
                         validation_data=self.valid_ds,
                         validation_steps=1,
                         callbacks=callbacks,
                         verbose=0)

        epoch_acc = hist.history['acc']
        return len(epoch_acc)

    def compare_optimizers_test_lookahead(self, use_wrapper=True):
        return  # don't run in CI as this takes time

        lr = 3e-3
        repeat_count = 10

        print("learning_rate", lr)
        print(" repeat_count", repeat_count)
        print("        eager", tf.executing_eagerly())
        print("         impl", "wrapper" if use_wrapper else "callback")

        for optimizer in [
                          pf.optimizers.RAdam(learning_rate=lr),
                          keras.optimizers.Adam(learning_rate=lr),
                          keras.optimizers.RMSprop(learning_rate=lr),
                          keras.optimizers.SGD(learning_rate=lr)]:
            for use_lookahead in [True, False]:
                epochs = [self.train_model(optimizer, use_lookahead, use_wrapper) for _ in range(repeat_count)]
                epochs = np.array(epochs)
                print("(opt:{:>8s}, LA:{:>6s} impl:{:>9s}) trained in {:4.1f} epochs (std: {:.2f})".format(
                    optimizer.__class__.__name__, str(use_lookahead),
                    "wrapper" if use_wrapper else "callback",
                    epochs.mean(), epochs.std()))

    def test_lookahead_callback(self):
        self.compare_optimizers_test_lookahead(use_wrapper=False)

    def test_lookahead_wrapper(self):
        self.compare_optimizers_test_lookahead(use_wrapper=True)

