# coding=utf-8
#
# created by kpe on 04.Sep.2019 at 15:24
#

from __future__ import absolute_import, division, print_function


import unittest

from tensorflow import keras

import params_flow as pf


class TestModelRegularization(unittest.TestCase):

    def test_add_dense_loss(self):
        model = keras.models.Sequential([
            keras.layers.Dense(2),
            keras.layers.Softmax()
        ])
        model.build(input_shape=(None, 2))
        pf.utils.add_dense_layer_loss(model)

    def test_cover(self):
        # coverage only
        model = keras.models.Sequential([
            keras.layers.Dense(2),
            keras.layers.Softmax()
        ])
        model.build(input_shape=(None, 2))
        pf.utils.add_dense_layer_loss(model, None, None)
