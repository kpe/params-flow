# coding=utf-8
#
# created by kpe on 20.Mar.2019 at 16:41
#

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import params as pp
import params_flow as pf


class Model(pp.WithParams, tf.keras.Model):
    class Params(pp.WithParams.Params):
        pass

    def _construct(self, *args, **kwargs):
        """ Override model construction. """
        super()._construct(*args, **kwargs)

    @property
    def params(self):
        return self._params

    def compute_mask(self, inputs, mask):
        return mask  # pragma: no cover
