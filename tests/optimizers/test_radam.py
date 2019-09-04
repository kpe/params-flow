# coding=utf-8
#
# created by kpe on 16.Aug.2019 at 14:58
#

from __future__ import absolute_import, division, print_function

import unittest

import tensorflow as tf

from tensorflow import keras

import params_flow as pf


class ModelRAdam(unittest.TestCase):

    def test_seq_model(self):
        model = keras.Sequential([
            keras.layers.Dense(17),
        ])
        model.compute_output_shape(input_shape=(16, 3, 4))
        # model.build(input_shape=(16, 3, 4))
        model.compile(optimizer=pf.optimizers.RAdam(),
                      loss='mse')
        model.fit(tf.ones((16, 3, 4), dtype=tf.float32),
                  tf.ones((16, 3, 17), dtype=tf.float32),
                  steps_per_epoch=2,
                  epochs=20,
                  callbacks=[pf.utils.create_one_cycle_lr_scheduler(
                          max_learn_rate=5e-2,
                          end_learn_rate=1e-7,
                          warmup_epoch_count=5,
                          total_epoch_count=10)
                  ])
        model.summary()

