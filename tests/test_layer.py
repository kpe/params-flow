# coding=utf-8
#
# created by kpe on 08.04.2019 at 9:11 PM
#

from __future__ import division, absolute_import, print_function

import unittest
import tempfile

import os
import params as pp

import tensorflow as tf
import params_flow as pf
from params_flow import Layer, Model


class SomeLayer(Layer):
    class Params(Layer.Params):
        center = True
        scale  = True


class SomeModel(Model):
    class Params(Model.Params):
        center = True
        scale  = True


class ParamsFlowTest(unittest.TestCase):

    def test_params_flow_layer_params(self):
        slayer = SomeLayer()
        self.assertTrue(slayer.params is not None)
        self.assertEqual(slayer.params.center, True)
        self.assertEqual(slayer.params.scale, True)
        self.assertEqual(slayer.get_config()['center'], True)
        self.assertEqual(slayer.get_config()['scale'], True)

        slayer = SomeLayer(center=False, scale=False)
        self.assertTrue(slayer.params is not None)
        self.assertEqual(slayer.params.center, False)
        self.assertEqual(slayer.params.scale, False)
        self.assertEqual(slayer.get_config()['center'], False)
        self.assertEqual(slayer.get_config()['scale'], False)

        slayer = SomeLayer.from_config(slayer.get_config())
        self.assertEqual(slayer.params.center, False)
        self.assertEqual(slayer.params.scale, False)
        self.assertEqual(slayer.get_config()['center'], False)
        self.assertEqual(slayer.get_config()['scale'], False)

    def test_serialization(self):
        slayer = SomeLayer()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "params.json")
            with open(temp_file, "wt") as f:
                f.write(slayer.params.to_json_string())
            nlayer = SomeLayer.from_json_file(temp_file)

        self.assertEqual(dict(slayer.params), dict(nlayer.params))

        nlayer = SomeLayer.from_params(slayer.params)
        self.assertEqual(dict(slayer.params), dict(nlayer.params))

    def test_shape_list(self):
        sh = SomeLayer.get_shape_list(tf.keras.Input(shape=(2, 3)))
        self.assertEqual(2, sh[1])
        self.assertEqual(3, sh[2])

    def test_model(self):
        smodel = SomeModel(center=True, scale=True)
        self.assertEqual(smodel.params, SomeModel().params)

    def test_layer(self):
        slayer = SomeLayer(center=True, scale=True)
        self.assertEqual(slayer.params, SomeLayer().params)

