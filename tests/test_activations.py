# coding=utf-8
#
# created by kpe on 22.May.2019 at 11:48
#

from __future__ import absolute_import, division, print_function

import unittest

import tensorflow as tf
import numpy as np


import params_flow as pf


class TestActivations(unittest.TestCase):

    def setUp(self) -> None:
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.enable_eager_execution()

    def test_gelu(self):
        gelu = pf.get_activation("gelu")
        gelu_exact = pf.get_activation("gelu_exact")
        self.assertTrue(np.allclose(gelu(0.5).numpy(), gelu_exact(0.5).numpy(), atol=1e-4))

    def test_get_activations(self):
        self.assertEqual(tf.nn.relu, pf.get_activation(tf.nn.relu))
        self.assertEqual(None, pf.get_activation(None))

        for act in ["linear", "relu", "tanh", "gelu", "gelu_exact", tf.nn.relu]:
            activation = pf.get_activation(act)
            self.assertTrue(activation is not None or act == "linear")
        try:
            pf.get_activation("non-existent")
        except ValueError:
            pass


