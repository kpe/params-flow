# coding=utf-8
#
# created by kpe on 10.08.2019 at 1:51 PM
#

from __future__ import division, absolute_import, print_function

import unittest
import tempfile

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

import params_flow as pf


class TestWrapper(unittest.TestCase):

    def test_wrappers(self):

        model = keras.models.Sequential([
            pf.wrappers.Concat([
                keras.layers.GlobalAveragePooling1D(),
                keras.layers.GlobalMaxPool1D(),
            ])
        ])

        model.build(input_shape=(None, 4, 2))

        x = np.array([
                [[1, -2], [2, -10], [3, 10], [4, 2]]
        ], dtype=np.float32)

        res = model.predict(x)
        print(res)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "model")
            model.save(temp_file)
            model = keras.models.load_model(temp_file, custom_objects={
                "Concat": pf.Concat
            })
            model.build(input_shape=(None, 4, 2))
            model.summary()

        res = model.predict(x)
        print(res)

        # coverage
        res = model.layers[0].call(x)
        print(res)
        model = keras.models.Sequential.from_config(model.get_config(), custom_objects={"Concat": pf.Concat})

    def test_compile(self):
        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(11, 4)),
            keras.layers.TimeDistributed(keras.layers.Dense(10)),
            pf.wrappers.Concat([
                keras.layers.GlobalAveragePooling1D(),
                keras.layers.GlobalMaxPool1D(),
            ]),
            keras.layers.Dense(2)
        ])
        model.build(input_shape=(None, 11, 4))
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[keras.metrics.SparseCategoricalAccuracy()])

        # coverage
        res = pf.wrappers.Concat([
                keras.layers.GlobalAveragePooling1D(),
                keras.layers.GlobalMaxPool1D(),
            ]).compute_output_shape((None, 11, 4, None))
        print(res)
