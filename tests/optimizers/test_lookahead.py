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



    @staticmethod
    def to_train_pair(x):
        y = tf.reduce_sum(x, axis=-1)
        y = tf.mod(y, 2)
        return tf.cast(x, tf.float32), y

    @staticmethod
    def get_ds(shape):
        ds_shape = [1000]+list(shape)
        def gen():
            while True:
                x = np.round(np.random.random_sample(ds_shape)).astype(np.int32)
                yield x
        ds = tf.data.Dataset.from_generator(gen, output_shapes=ds_shape, output_types=tf.int32)
        ds = ds.apply(tf.data.experimental.unbatch())
        return ds

    def test_lookahead(self):
        return  # don't run in CI as this takes time

        ds = self.get_ds(shape=(16, 4))
        ds = ds.map(self.to_train_pair)
        ds = ds.shuffle(1000)

        for optimizer in [keras.optimizers.Adam(), pf.optimizers.RAdam(),
                          keras.optimizers.RMSprop(), keras.optimizers.SGD()]:
            for use_lookahead in [True, False]:
                callbacks = [pf.optimizers.LookaheadOptimizerCallback()] if use_lookahead else []
                epochs = [self.train_model(ds, optimizer, callbacks) for _ in range(10)]
                epochs = np.array(epochs)
                print("(opt:{:>8s}, LA:{:>6s}) trained in {:4.1f} epochs (std: {:.2f})".format(
                    optimizer.__class__.__name__, str(use_lookahead),
                    epochs.mean(), epochs.std()))

    class StopTraining(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            acc = logs.get("acc")
            if np.abs(acc - 1.) < 0.0001:
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
                         epochs=500,
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

    def test_cover_wrap(self):
        ds = self.get_ds(shape=(2, 4))
        ds = ds.repeat()
        ds = ds.map(self.to_train_pair)
        model = keras.Sequential([
            keras.layers.Dense(4, activation="tanh"),
            keras.layers.Dense(2, activation="softmax"),
        ])
        model.build(input_shape=(2, 4))
        model.compile(optimizer=pf.optimizers.RAdam(),
                      loss=keras.losses.SparseCategoricalCrossentropy(),)

        #pf.optimizers.lookahead.OptimizerLookaheadWrapper().wrap(model)

        model.fit(ds, steps_per_epoch=1, epochs=20)

    def train_model_wrap(self, ds, optimizer, use_lookahead):

        model = keras.Sequential([
            keras.layers.Dense(32, activation="tanh"),
            keras.layers.Dense(2, activation="softmax"),
        ])

        model.compile(optimizer=optimizer,
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

        if use_lookahead:
            lookahead = pf.optimizers.lookahead.OptimizerLookaheadWrapper()
            model = lookahead.wrap(model)

        hist = model.fit(ds,
                         steps_per_epoch=200,
                         epochs=500,
                         callbacks=[self.StopTraining()],
                         verbose=0)
        epoch_acc = hist.history['acc']
        return len(epoch_acc)

    def test_lookahead_wrap(self):
        return  # don't run in CI as this takes time

        ds = self.get_ds(shape=(16, 4))
        ds = ds.map(self.to_train_pair)
        ds = ds.shuffle(1000)

        for optimizer in [keras.optimizers.Adam(), pf.optimizers.RAdam(),
                          keras.optimizers.RMSprop(), keras.optimizers.SGD()]:
            for use_lookahead in [True, False]:
                epochs = [self.train_model_wrap(ds, optimizer, use_lookahead) for _ in range(10)]
                epochs = np.array(epochs)
                print("(opt:{:>8s}, LA:{:>6s}) trained in {:4.1f} epochs (std: {:.2f})".format(
                    optimizer.__class__.__name__, str(use_lookahead),
                    epochs.mean(), epochs.std()))

