# coding=utf-8
#
# created by kpe on 05.Sep.2019 at 16:36
#

from __future__ import absolute_import, division, print_function


import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python import keras

import params_flow as pf


class LookAheadTest(unittest.TestCase):

    def setUp(self) -> None:
        tf.enable_eager_execution()

    def create_data_gen(self, shape=(10, )):
        def gen():
            x = tf.random_uniform(shape, 0, 2, dtype=tf.int32)
            yield x
        return gen

    @staticmethod
    def to_train_pair(x):
        y = tf.reduce_sum(x, axis=-1)
        y = tf.mod(y, 2)
        return tf.cast(x, tf.float32), y

    def get_ds(self, shape):
        ds = tf.data.Dataset.from_generator(self.create_data_gen(shape),
                                            output_shapes=shape,
                                            output_types=tf.int32)
        return ds

    def test_lookahead(self):
        return  # don't run in CI as this takes time

        ds = self.get_ds(shape=(128, 4))
        ds = ds.repeat()
        ds = ds.map(self.to_train_pair)

        for optimizer in [keras.optimizers.Adam(), pf.optimizers.RAdam(),
                          keras.optimizers.RMSprop(), keras.optimizers.SGD()]:
            for use_lookahead in [True, False]:
                callbacks = [pf.optimizers.LookaheadOptimizerCallback()] if use_lookahead else []
                epochs = [self.train_model(ds, optimizer, callbacks) for _ in range(10)]
                epochs = np.array(epochs)
                print("(opt:{:>8s}, LA:{:>6s}) trained in {:4.1f} epochs (std: {:.1f})".format(
                    optimizer.__class__.__name__, str(use_lookahead),
                    epochs.mean(), epochs.std()))

    class StopTraining(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if tf.math.abs(logs.get("acc") - 1.) < 0.0001:
                self.model.stop_training = True

    def train_model(self, ds, optimizer, callbacks):

        model = keras.Sequential([
            keras.layers.Dense(32, activation="tanh"),
            keras.layers.Dense(2, activation="softmax"),
        ])

        model.compile(optimizer=optimizer,
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

        hist = model.fit(ds,
                         steps_per_epoch=200,
                         epochs=100,
                         callbacks=callbacks+[self.StopTraining()],
                         verbose=0)
        epoch_acc = hist.history['acc']
        return len(epoch_acc)

    def test_cover(self):
        ds = self.get_ds(shape=(2, 4))
        ds = ds.repeat()
        ds = ds.map(self.to_train_pair)
        model = keras.Sequential([
            keras.layers.Dense(4, activation="tanh"),
            keras.layers.Dense(2, activation="softmax"),
        ])
        model.compile(optimizer=pf.optimizers.RAdam(), loss=keras.losses.SparseCategoricalCrossentropy(),)
        model.fit(ds, steps_per_epoch=1, epochs=20,
                  callbacks=[pf.optimizers.LookaheadOptimizerCallback()])
