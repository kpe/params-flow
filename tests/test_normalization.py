# coding=utf-8
#
# created by kpe on 08.04.2019 at 9:12 PM
#

from __future__ import division, absolute_import, print_function

import tempfile
import unittest

import os
import numpy as np
from tensorflow import keras

from params_flow import LayerNormalization


class TestLayerNormalization(unittest.TestCase):

    def test_simple(self):

        norm_layer = LayerNormalization()
        model = keras.Sequential([keras.layers.InputLayer(input_shape=(2,3)),
                                  norm_layer])

        model.build(input_shape=(None, 2, 3))
        model.compile(optimizer=keras.optimizers.Adam(), loss='mse')
        model.summary()

        inputs = np.array([[
            [.2, .1, .3],
            [.5, .1, .1]
        ]])

        predict = model.predict(inputs)

        expected = np.asarray([[
            [0,      -1.2247,  1.2247],
            [1.4142, -0.7071, -0.7071]
        ]])

        self.assertTrue(np.allclose(predict, expected, atol=1e-4))

    def test_equal(self):
        norm_layer = LayerNormalization()
        model = keras.Sequential([keras.layers.InputLayer(input_shape=(16, 256)),
                                  norm_layer])

        # model.build(input_shape=(3, 16, 256))
        model.compile(optimizer=keras.optimizers.Adam(), loss='mse')
        # model.summary()

        model.fit(np.zeros((3,16,256)), np.ones((3,16,256)))
        model.summary()

        inputs = np.zeros((3, 16, 256))
        predicted = model.predict(inputs)
        expected  = np.ones_like(inputs)
        np.allclose(expected, predicted)

    def test_serialization(self):
        model = keras.Sequential([
            LayerNormalization(input_shape=(2, 3))
        ])
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "model")
            model.save(temp_file)
            model = keras.models.load_model(temp_file, custom_objects={
                "LayerNormalization": LayerNormalization
            })
            model.summary()

        encoded = model.to_json()
        model = keras.models.model_from_json(encoded, custom_objects={
            "LayerNormalization": LayerNormalization
        })
        model.summary()

