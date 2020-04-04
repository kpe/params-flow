# coding=utf-8
#
# created by kpe on 15.Mar.2019 at 11:11
#

from __future__ import absolute_import, division, print_function

import json

import tensorflow as tf
import params as pp
import params_flow as pf

from params import Params


class Layer(pp.WithParams, tf.keras.layers.Layer):
    class Params(pp.WithParams.Params):
        pass

    def _construct(self, **kwargs):
        """ Override layer construction. """
        super()._construct(**kwargs)

    @property
    def params(self) -> Params:
        return self._params

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape  # pragma: no cover

    def get_config(self):
        base_config = super(Layer, self).get_config()
        return dict(list(base_config.items()) + list(self.params.items()))

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `Layer` from a json file of parameters."""
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls(**json.loads(text))

    @staticmethod
    def get_shape_list(tensor):
        """ Tries to return the static shape as a list
        falling back to dynamic shape per dimension. """
        static_shape, dyn_shape = tensor.shape.as_list(), tf.shape(tensor)

        def shape_dim(ndx):
            return dyn_shape[ndx] if static_shape[ndx] is None else static_shape[ndx]

        shape = map(shape_dim, range(tensor.shape.ndims))
        return list(shape)
