# coding=utf-8
#
# created by kpe on 09.08.2019 at 10:00 PM
#

from __future__ import division, absolute_import, print_function

from tensorflow import keras


def flatten_layers(keras_layer):
    if isinstance(keras_layer, keras.layers.Layer):
        yield keras_layer
    for keras_layer in keras_layer._layers:
        for sub_layer in flatten_layers(keras_layer):
            yield sub_layer


def freeze_leaf_layers(layer_or_model, freeze_selector_fn):
    """
    Freezes all leaf layers selected by the given freeze_selector_fn
    and unfreezes all other leaf layers.
    """
    for layer in flatten_layers(layer_or_model):
        # freeze leafs only - as trainable is forced on children
        if len(layer._layers) == 0:
            layer.trainable = not freeze_selector_fn(layer)
