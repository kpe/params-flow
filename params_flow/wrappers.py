# coding=utf-8
#
# created by kpe on 10.08.2019 at 12:25 PM
#

from __future__ import division, absolute_import, print_function

import copy
from typing import List

import tensorflow as tf
from tensorflow import keras

from params_flow import Layer


class Wrapper(Layer):
    class Params(Layer.Params):
        pass

    def __init__(self, layers: List[keras.layers.Layer], **kwargs):
        super(Wrapper, self).__init__(**kwargs)
        self.layers = [layer for layer in layers]

    def get_config(self):
        base_config = super(Wrapper, self).get_config()
        layer_configs = []
        for layer in self.layers:
            layer_configs.append({
                "class_name": layer.__class__.__name__,
                "config": layer.get_config()
            })
        config = {
            "config": copy.deepcopy(base_config),
            "layers": copy.deepcopy(layer_configs)
        }

        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        wrapped_layers = []
        for layer_config in config['layers']:
            layer = keras.layers.deserialize(layer_config,
                                             custom_objects=custom_objects)
            wrapped_layers.append(layer)
        wrapper = cls(wrapped_layers, **config['config'])
        return wrapper


class Concat(Wrapper):
    """
    A Keras Sequential model friendly wrapper concatenating the outputs
    of the given a list of layers in a single output.

    Example usage:

       model = Sequential([
          Concat([
             GlobalAveragePooling1D(),
             GlobalMaxPooling1D()
          ])
          Dense(2)
       ])

    """
    class Params(Wrapper.Params):
        axis = -1

    def call(self, inputs, **kwargs):
        outputs = [layer(inputs, **kwargs) for layer in self.layers]
        output = tf.concat(outputs, axis=self.params.axis)
        return output

