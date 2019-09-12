# coding=utf-8
#
# created by kpe on 05.Sep.2019 at 16:05
#

from __future__ import absolute_import, division, print_function


import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops


class LookaheadOptimizerCallback(keras.callbacks.Callback):
    """
    This class provides an implementation of
    the [Lookahead Optimizer](https://arxiv.org/abs/1907.08610)
    implemented as a Keras Callback, by keeping track of the slow weights
    and updating the model weights on every k-th batch end.
    """

    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = 0
        self.slow_weights = None

    def on_train_batch_end(self, batch, logs=None):
        self.count += 1
        if self.slow_weights is None:
            with tf.control_dependencies(self.model.trainable_weights):
                self.slow_weights = []
                for fast_param in self.model.trainable_weights:
                    with ops.control_dependencies([fast_param]):
                        slow_param = tf.Variable(fast_param.initialized_value(),
                                                 dtype=fast_param.dtype,
                                                 trainable=False,
                                                 name=fast_param.name.split(":")[0])
                    self.slow_weights.append(slow_param)
                    K.track_variable(slow_param)
        else:
            if self.count % self.k == 0:
                slow_ups, fast_ups = [], []
                for fast, slow in zip(self.model.trainable_weights,
                                      self.slow_weights):
                    slow_ups.append(K.update(slow, slow + self.alpha * (fast - slow)))
                with tf.control_dependencies(slow_ups):
                    for fast, slow in zip(self.model.trainable_weights,
                                          self.slow_weights):
                        fast_ups.append(K.update(fast, slow))
                K.batch_get_value(slow_ups)
                K.batch_get_value(fast_ups)


class OptimizerLookaheadWrapper:
    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = None
        self.slow_weights = None

    def wrap(self, model: keras.models.Model):
        with K.name_scope("training"):
            with K.name_scope("Lookahead"):
                # initialize counter and slow_weights
                self.count = tf.Variable(0, dtype=tf.int32, trainable=False, name="update_count")
                K.track_variable(self.count)

                self.slow_weights = []
                for fast_param in model.trainable_weights:
                    with ops.control_dependencies([fast_param]):
                        slow_param = tf.Variable(fast_param.initialized_value(),
                                                 dtype=fast_param.dtype,
                                                 trainable=False,
                                                 name=fast_param.name.split(":")[0])
                    self.slow_weights.append(slow_param)
                    K.track_variable(slow_param)

        def lookahead_update():
            with K.name_scope("training"):
                with K.name_scope("Lookahead"):

                    # count++ mod k
                    count_op = state_ops.assign_add(self.count, 1)

                    def fast_update():
                        return control_flow_ops.no_op()

                    def slow_update():
                        with ops.control_dependencies(model.trainable_weights):
                            slow_ups = [state_ops.assign_add(slow, (fast - slow) * self.alpha)
                                        for slow, fast in zip(self.slow_weights, model.trainable_weights)]

                        with ops.control_dependencies(slow_ups):
                            fast_ups = [state_ops.assign(fast, slow)
                                        for slow, fast in zip(self.slow_weights, model.trainable_weights)]

                        return control_flow_ops.group(*fast_ups)

                    with ops.control_dependencies([count_op]):
                        update_op = control_flow_ops.cond(math_ops.equal(math_ops.mod(self.count, self.k), 0),
                                                          slow_update, fast_update)
                    return update_op

        model.add_update(lookahead_update)

        return model
