# coding=utf-8
#
# created by kpe on 08.04.2019 at 8:57 PM
#

from __future__ import division, absolute_import, print_function

from params import Params
from tensorflow.python import keras


class Model(keras.Model):
    class Params(Params):
        pass

    def __init__(self, **kwargs):
        _params, other_args = self.__class__.Params.from_dict(kwargs)
        super(Model, self).__init__(**other_args)
        self._params = _params
        self._construct(self.params)

    @property
    def params(self):
        return self._params

    def compute_mask(self, inputs, mask):
        return mask  # pragma: no cover

    def _construct(self, params):
        """ Override model construction. """
        pass

