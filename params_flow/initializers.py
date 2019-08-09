# coding=utf-8
#
# created by kpe on 31.07.2019 at 8:06 PM
#

from __future__ import division, absolute_import, print_function


import tensorflow as tf
import params as pf


def get_initializer(params: pf.Params):
    if params.initializer == "linear" or params.initializer is None:
        initializer = None
    elif params.initializer == "uniform":
        initializer = tf.compat.v2.initializers.RandomUniform(
            minval=-params.initializer_range,
            maxval=params.initializer_range,
            seed=params.random_seed)
    elif params.initializer == "normal":
        initializer = tf.compat.v2.initializers.RandomNormal(
            stddev=params.initializer_range,
            seed=params.random_seed)
    elif params.initializer == "truncated_normal":
        initializer = tf.compat.v2.initializers.TruncatedNormal(
            stddev=params.initializer_range,
            seed=params.random_seed)
    else:
        raise ValueError("Initializer {} not supported".format(params.initializer))

    return initializer
