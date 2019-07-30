# coding=utf-8
#
# created by kpe on 15.Mar.2019 at 11:11
#

from __future__ import absolute_import, division, print_function

import json

import tensorflow as tf

from params import Params
from tensorflow.python import keras


def get_initializer(params: Params):
    if params.initializer == "uniform":
        initializer = tf.compat.v2.initializers.RandomUniform(minval=-params.initializer_range,
                                                              maxval=params.initializer_range,
                                                              seed=params.random_seed)
    elif params.initializer == "normal":
        initializer = tf.compat.v2.initializers.RandomNormal(stddev=params.initializer_range,
                                                             seed=params.random_seed)
    elif params.initializer == "truncated_normal":
        initializer = tf.compat.v2.initializers.TruncatedNormal(stddev=params.initializer_range,
                                                                seed=params.random_seed)
    else:
        raise ValueError("Initializer {} not supported".format(params.initializer))

    return initializer


class Layer(keras.layers.Layer):
    class Params(Params):
        trainable   = True
        name        = None
        dtype       = tf.float32.name
        dynamic     = False

    def __init__(self, **kwargs):
        self._params, other_args = self.__class__.Params.from_dict(kwargs)
        super(Layer, self).__init__(trainable=self._params.trainable,
                                    name=self._params.name,
                                    dtype=tf.dtypes.as_dtype(self._params.dtype),
                                    dynamic=self._params.dynamic,
                                    **other_args)
        self._construct(self.params)

    @property
    def params(self) -> Params:
        return self._params

    def _construct(self, params):
        """ Override layer construction. """
        pass

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

    @classmethod
    def from_params(cls, params_dict, **kwargs):
        """
        Creates an instance from the specified parameters (by overriding params_dict with kwargs).
        """
        layer_instance = cls(
            **cls.Params(
                cls.Params.from_dict(params_dict,                   # read relevant params from params_dict
                                     return_unused=False,
                                     return_instance=False),
                **kwargs                                            # override with kwargs
            )
        )
        return layer_instance
