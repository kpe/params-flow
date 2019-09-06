# coding=utf-8
#
# created by kpe on 05.Sep.2019 at 16:05
#

from __future__ import absolute_import, division, print_function


import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K


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
            self.slow_weights = self.model.trainable_weights
        else:
            if self.count % self.k == 0:
                slow_ups, fast_ups = [], []
                for fast, slow in zip(self.model.trainable_weights,
                                      self.slow_weights):
                    slow_ups.append(K.update(slow, slow + self.alpha * (fast - slow)))
                    fast_ups.append(K.update(fast, slow))
                K.batch_get_value(slow_ups)
                K.batch_get_value(fast_ups)


'''
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops


class OptimizerLookaheadWrapper:
    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = None
        self.slow_weights = None

    def wrap(self, model: keras.models.Model):
        _orig_make_train_fn = model._make_train_function

        def _make_train_function():
            _orig_make_train_fn()

            if not hasattr(model, "train_function"):
                raise RuntimeError("Call Model.compile() first.")

            train_function = model.train_function

            with K.name_scope("training"):
                with K.name_scope("Lookahead"):
                    self.count = K.variable(0, dtype=tf.int32, name="update_count")
                    self.slow_weights = [K.variable(fast_param.value(), dtype=fast_param.dtype)
                                         for fast_param in model.trainable_weights]

                    count_op = control_flow_ops.cond(tf.equal(self.count, self.k),
                                                     lambda: tf.assign(self.count, 0),
                                                     lambda: tf.assign_add(self.count, 1))

                    def fast_update():
                        return control_flow_ops.no_op()

                    def slow_update():
                        slow_ups  = [state_ops.assign_add(slow, (fast - slow) * self.alpha)
                                     for slow, fast in zip(self.slow_weights, model.trainable_weights)]

                        with tf.control_dependencies(slow_ups):
                            slow_ups += [state_ops.assign(fast, slow)
                                         for slow, fast in zip(self.slow_weights, model.trainable_weights)]

                        return control_flow_ops.group(*slow_ups)

                    with tf.control_dependencies([count_op] + [train_function.updates_op]):
                        update_op = tf.cond(tf.equal(self.count, self.k), slow_update, fast_update)

                    wrapper_ups = [update_op]
                    wrapper_fn = K.function(inputs=train_function.inputs,
                                            outputs=train_function.outputs,
                                            updates=[train_function.updates_op] + wrapper_ups,
                                            **train_function.session_kwargs)

            model.train_function = wrapper_fn

        if model._make_train_function == _make_train_function:
            raise RuntimeError("Already wrapped")

        model._make_train_function = _make_train_function
        return model
'''

