# coding=utf-8
#
# created by kpe on 08.04.2019 at 9:07 PM
#

from __future__ import division, absolute_import, print_function

import tensorflow as tf
from tensorflow.python import keras

from params_flow.layer import Layer


class LayerNormalization(Layer):
    """
    Layer normalization layer from arXiv:1607.06450.
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        See: tf.contrib.layers.layer_norm
    """
    class Params(Layer.Params):
        epsilon         = 1e-12

    def _construct(self, params):
        self.gamma = None
        self.beta  = None
        self.supports_masking = True

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)
        self.gamma = self.add_weight(name="gamma", shape=input_shape[-1:], initializer=keras.initializers.Ones(),
                                     trainable=True)
        self.beta  = self.add_weight(name="beta", shape=input_shape[-1:], initializer=keras.initializers.Zeros(),
                                     trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):                               # pragma: no cover
        x = inputs
        if tf.__version__.startswith("2."):
            mean, var = tf.nn.moments(x, axes=-1, keepdims=True)
        else:
            mean, var = tf.nn.moments(x, axes=-1, keep_dims=True)
        # normed = (x - mean)/tf.sqrt(var + self.params.epsilon)
        # res    = self.gamma * normed + self.beta
        res = tf.nn.batch_normalization(x, mean=mean, variance=var,
                                        scale=self.gamma,
                                        offset=self.beta,
                                        variance_epsilon=self.params.epsilon)
        return res
