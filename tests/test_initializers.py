# coding=utf-8
#
# created by kpe on 31.07.2019 at 8:20 PM
#

from __future__ import division, absolute_import, print_function


import unittest

import params
import params_flow as pf


class TestInitializers(unittest.TestCase):
    def test_initializers(self):
        class IParams(params.Params):
            initializer_range = 0.01
            random_seed       = None
            initializer       = "normal"

        self.assertIsNotNone(pf.get_initializer(IParams(initializer="normal")))
        self.assertIsNotNone(pf.get_initializer(IParams(initializer="truncated_normal")))
        self.assertIsNotNone(pf.get_initializer(IParams(initializer="uniform")))
        self.assertIsNone(pf.get_initializer(IParams(initializer="linear")))
        self.assertIsNone(pf.get_initializer(IParams(initializer=None)))
        try:
            pf.get_initializer(IParams(initializer="non-existing"))
            self.fail()
        except ValueError:
            pass

