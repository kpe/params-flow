# coding=utf-8
#
# created by kpe on 15.Mar.2019 at 11:25
#

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python import keras

from params_flow.layer import Layer


class Normalization(Layer):
    class Params(Layer.Params):
        pass

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask


class LayerNormalization(Normalization):
    """
    Layer normalization layer from arXiv:1607.06450.
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        See: https://github.com/CyberZHG/keras-layer-normalization
        See: tf.contrib.layers.layer_norm
    """
    class Params(Normalization.Params):
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

        #
        # this is how we would normalize, but
        #    it's commented out as it is not numerically equivalent
        #    to the tf.nn.batch_normalization implementation (used in BERT)
        #
        # normed = (x - mean)/tf.sqrt(var + self.params.epsilon)
        # res    = self.gamma * normed + self.beta
        # res = tf.nn.batch_normalization(x, mean=mean, variance=var,
        #                                 scale=self.gamma,
        #                                 offset=self.beta,
        #                                 variance_epsilon=self.params.epsilon)
        #

        # following two lines represent the tf.nn.batch_normalization implementation
        inv = self.gamma * tf.math.rsqrt(var + self.params.epsilon)
        res = x * tf.cast(inv, x.dtype) + tf.cast(self.beta - mean * inv, x.dtype)

        return res
