# coding=utf-8
#
# created by kpe on 10.08.2019 at 12:39 AM
#

from __future__ import division, absolute_import, print_function

import math

from tensorflow import keras


def create_one_cycle_lr_scheduler(max_learn_rate=5e-5,
                                  end_learn_rate=1e-7,
                                  warmup_epoch_count=10,
                                  total_epoch_count=90):

    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate / warmup_epoch_count) * (epoch + 1)
        else:
            k = math.log(end_learn_rate / max_learn_rate) / (total_epoch_count - warmup_epoch_count)
            res = max_learn_rate * math.exp(k * (epoch - warmup_epoch_count))
        return float(res)

    learning_rate_scheduler = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler
