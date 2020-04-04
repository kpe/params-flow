# coding=utf-8
#
# created by kpe on 10.08.2019 at 12:25 PM
#

from __future__ import division, absolute_import, print_function

import copy
from typing import List

import tensorflow as tf
import params_flow as pf


class Wrapper(pf.Layer):
    class Params(pf.Layer.Params):
        pass

    def _construct(self, layers: List[tf.keras.layers.Layer], **kwargs):
        super()._construct(**kwargs)
        self.layers = [layer for layer in layers]
        self.supports_masking = True

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
            layer = tf.keras.layers.deserialize(layer_config, custom_objects=custom_objects)
            wrapped_layers.append(layer)
        wrapper = cls(wrapped_layers, **config['config'])
        return wrapper

    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
        for layer in self.layers:
            layer.build(input_shape)
        super(Wrapper, self).build(input_shape)


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

    def compute_output_shape(self, input_shape):
        output_shapes = [layer.compute_output_shape(input_shape).as_list()
                         for layer in self.layers]
        assert min([len(sh) for sh in output_shapes]) == max([len(sh) for sh in output_shapes])
        ndims = max([len(sh) for sh in output_shapes])

        out_shape = []
        for ndx, dim_shapes in enumerate(zip(*output_shapes)):
            if ndx == (self.params.axis + ndims) % ndims:
                out_shape.append(sum(dim_shapes))
            else:
                dim_shape = None
                for sh in dim_shapes:
                    if dim_shape is None:
                        dim_shape = sh
                    else:
                        assert dim_shape == sh  # pragma: no cover
                out_shape.append(dim_shape)

        return out_shape

    def call(self, inputs, **kwargs):
        outputs = [layer(inputs, **kwargs) for layer in self.layers]
        output = tf.concat(outputs, axis=self.params.axis)
        return output

    def compute_mask(self, inputs, mask=None):
        return None
