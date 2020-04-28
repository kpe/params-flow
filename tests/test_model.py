# coding=utf-8
#
# created by kpe on 09.04.2019 at 5:11 AM
#

from __future__ import division, absolute_import, print_function

import unittest

import tensorflow as tf
import params_flow as pf
from params_flow import Layer, Model, LayerNormalization


class CustomLayer(Layer):
    class Params(Layer.Params):
        num_units = 11

    def _construct(self):
        super()._construct()
        self.supports_masking = True

    def build(self, input_shape):
        out_dims = self.params.num_units
        in_dims = input_shape[-1]
        shape = tf.TensorShape((out_dims, in_dims))
        self.kernel = self.add_weight(name="kernel",
                                      shape=shape,
                                      dtype=tf.float32,
                                      initializer="uniform",
                                      trainable=True)
        super(CustomLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.params.num_units
        return shape

    def call(self, inputs, training=None, mask=None):
        result = tf.tensordot(inputs, self.kernel, axes=[[-1], [-1]])
        return result


class CustomModel(Model):
    class Params(Model.Params):
        num_units = 12

    def _construct(self):
        super()._construct()
        self.layer = CustomLayer.from_params(self.params)
        self.norm  = LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, training=None, mask=None):
        res = self.layer(inputs, training=training, mask=mask)
        res = self.norm(res)
        return res


class ModelTest(unittest.TestCase):

    def test_model(self):
        model = CustomModel(num_units=13)
        # model.build(input_shape=(16,3,4))
        model.compile(optimizer='adam', loss='mse')
        model.fit(tf.zeros((16, 3, 4), dtype=tf.float32),
                  tf.ones((16, 3, 13), dtype=tf.float32), steps_per_epoch=2)
        model.summary()

    def test_seq_model(self):
        model = tf.keras.Sequential([CustomLayer(num_units=17),
                                     LayerNormalization()])
        model.compute_output_shape(input_shape=(16, 3, 4))
        # model.build(input_shape=(16, 3, 4))
        model.compile(optimizer='adam', loss='mse')
        model.fit(tf.ones((16, 3, 4), dtype=tf.float32),
                  tf.ones((16, 3, 17), dtype=tf.float32), steps_per_epoch=2, epochs=10,
                  callbacks=[pf.utils.create_one_cycle_lr_scheduler(
                          max_learn_rate=5e-2,
                          end_learn_rate=1e-7,
                          warmup_epoch_count=5,
                          total_epoch_count=10)
                  ])
        model.summary()

    def test_layer_as_model(self):
        model = CustomLayer(num_units=17).as_model(input_shape=(16, 3, 4))
        model.summary()
