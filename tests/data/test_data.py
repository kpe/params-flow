# coding=utf-8
#
# created by kpe on 28.04.2020 at 11:19 AM
#

from __future__ import division, absolute_import, print_function


import unittest

from typing import List, Text, Dict

import params_flow as pf
import tensorflow as tf
import numpy as np


class ADataSet(pf.BaseDataSet):
    class Params(pf.BaseDataSet.Params):
        param_a = 3
        param_b = 7
        num_examples = 12
        seq_len = 5

    def construct_data_items(self) -> List:
        res =  [[edx + (ndx * self.params.param_b) % self.params.param_a
                 for ndx in range(self.params.seq_len)]
                for edx in range(self.params.num_examples)]
        return res #[tf.data.Dataset.from_tensor_slices({"feature": d}) for d in res]


class InFeature(pf.Feature):
    class Params(pf.Feature.Params):
        output_feature = "feature"

    def get_feature_transform(self):
        @tf.function
        def mean_feature_transform(feature) -> Dict[Text, tf.Tensor]:
            return {self.params.output_feature: feature}
        return mean_feature_transform


class MeanFeature(pf.Feature):
    class Params(pf.Feature.Params):
        input_feature  = "feature"
        output_feature = "mean"

    def get_feature_transform(self):
        @tf.function
        def mean_feature_transform(example: Dict) -> Dict[Text, tf.Tensor]:
            feature = example[self.params.input_feature]

            mean = tf.reduce_mean(feature)

            result = {key: val for key, val in example.items()}
            result.update({self.params.output_feature: mean})
            return result
        return mean_feature_transform


class JoinFeature(pf.Feature):
    def transform_dataset(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        ds = ds.batch(2)

        @tf.function
        def concat_batches(example):
            feature = example["feature"]
            feature = tf.reshape(feature, (1, -1))
            result = dict(example)
            result.update({"feature": feature})
            return result
        ds = ds.map(concat_batches).unbatch()

        return ds


class AFeaturizer(pf.Featurizer):
    def featurize(self, ds: pf.BaseDataSet, train=True) -> pf.BaseDataSet:
        ds = ds.with_feature(InFeature())
        ds = ds.with_cache("/tmp/pf.cache")
        ds = ds.with_cache("")
        ds = ds.with_dataset_transform(JoinFeature())
        ds = ds.with_feature(MeanFeature())
        ds = ds.with_cache()
        ds = ds.with_keep_features(["mean", "feature"])
        ds = ds.with_drop_features([""])
        return ds


class TestDataFeaturizer(unittest.TestCase):

    def test_data(self):
        ads = ADataSet()
        ads = AFeaturizer().featurize(ads)
        train_ds, test_ds = ads.split(train_to_validation_ratio=1)
        train_ds, test_ds = ads.split([1, 1])

        print("size:", ads.get_size())

        for ndx, ex in enumerate(train_ds.build_dataset()):
            for key, val in ex.items():
                print(f"{ndx} {key:>10s}:", val.numpy())
        for ndx, ex in enumerate(test_ds.build_dataset(interleave=False)):
            for key, val in ex.items():
                print(f"{ndx} {key:>10s}", val.numpy())
