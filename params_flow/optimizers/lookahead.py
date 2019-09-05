# coding=utf-8
#
# created by kpe on 05.Sep.2019 at 16:05
#

from __future__ import absolute_import, division, print_function


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



