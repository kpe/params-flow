# coding=utf-8
#
# created by kpe on 04.Sep.2019 at 15:06
#

from __future__ import absolute_import, division, print_function

from typing import Union
from tensorflow import keras


def add_dense_layer_loss(model_or_layer: Union[keras.layers.Layer, keras.models.Model],
                         kernel_regularizer=keras.regularizers.l2(0.01),
                         bias_regularizer=keras.regularizers.l2(0.01)):
    add_model_loss(model_or_layer,
                   lambda layer: layer.kernel if isinstance(layer, keras.layers.Dense) else None,
                   kernel_regularizer)
    add_model_loss(model_or_layer,
                   lambda layer: layer.bias if isinstance(layer, keras.layers.Dense) else None,
                   bias_regularizer)


def add_model_loss(model_or_layer: Union[keras.layers.Layer, keras.models.Model],
                   weight_accessor_fn,
                   regularizer=keras.regularizers.l2(0.01)):

    layers = model_or_layer.layers if isinstance(model_or_layer, keras.models.Model) else model_or_layer._layers

    if regularizer is not None:
        for layer in layers:
            weight = weight_accessor_fn(layer)
            if weight is not None:
                if weight.trainable:
                    layer.add_loss(lambda: regularizer(weight))
