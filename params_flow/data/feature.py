# coding=utf-8
#
# created by kpe on 28.04.2020 at 11:08 AM
#

from __future__ import division, absolute_import, print_function


import params as pp
import tensorflow as tf


class Feature(pp.WithParams):
    class Params(pp.WithParams.Params):
        input_feature  = None
        output_feature = None

    def _construct(self):
        super()._construct()

    @property
    def params(self) -> Params:
        return self._params

    def get_feature_transform(self):
        raise RuntimeError("Not implemented")  # pragma: no cover

    def transform_dataset(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        raise RuntimeError("Not implemented")  # pragma: no cover

