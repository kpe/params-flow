# coding=utf-8
#
# created by kpe on 10.08.2019 at 1:51 PM
#

from __future__ import division, absolute_import, print_function

import unittest
import tempfile

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

        model.build(input_shape=(None, 3, 2))

        x = np.array([
                [[1, -2], [2, -10], [3, 10], [4, 2]]
        ], dtype=np.float32)

        res = model.predict(x)
        print(res)

        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.file.close()
            model.save(temp_file.name)
            model = keras.models.load_model(temp_file.name, custom_objects={
                "Concat": pf.Concat
            })
            model.summary()

        res = model.predict(x)
        print(res)

        # coverage
        res = model.layers[0].call(x)
        print(res)
