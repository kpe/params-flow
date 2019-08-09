# coding=utf-8
#
# created by kpe on 09.08.2019 at 10:08 PM
#

from __future__ import division, absolute_import, print_function

import unittest

import params_flow as pf

from tensorflow import keras


class FreezeTest(unittest.TestCase):

    def test_freeze(self):
        model = keras.models.Sequential([
            keras.layers.TimeDistributed(keras.layers.Dense(10)),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(10, name="frozen")
        ])

        model.build(input_shape=(5, 21, 3))

        trainable_count = len(model.trainable_weights)
        pf.utils.freeze_leaf_layers(model, lambda layer: layer.name == "frozen")
        frozen_count = trainable_count - len(model.trainable_weights)

        model.summary()

        self.assertEqual(frozen_count, 2)
