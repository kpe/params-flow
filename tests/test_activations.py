# coding=utf-8
#
# created by kpe on 22.May.2019 at 11:48
#

from __future__ import absolute_import, division, print_function

import unittest

import tensorflow as tf
import numpy as np


from params_flow import gelu, gelu_exact

tf.enable_eager_execution()


class TestActivations(unittest.TestCase):

    def test_gelu(self):
        self.assertTrue(np.allclose(gelu(0.5).numpy(), gelu_exact(0.5).numpy(), atol=1e-4))
