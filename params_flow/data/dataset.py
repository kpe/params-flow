# coding=utf-8
#
# created by kpe on 28.04.2020 at 11:08 AM
#

from __future__ import division, absolute_import, print_function


import os
import random
import pprint
from typing import Tuple, List, Text, Dict, Union, Any

from absl import logging as log

import numpy as np
import tensorflow as tf

import params as pp

from .feature import Feature


class _FeatureTransform:
    def transform(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        raise RuntimeError("Not implemented")    # pragma: no cover

    def _map_ds(self, fn, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.map(fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


class _InterleaveTransform(_FeatureTransform):
    def __init__(self, item_to_dataset, cycle_length=1, block_length=1):
        self.item_to_dataset = item_to_dataset
        self.cycle_length = cycle_length  # should be bigger than the batch_size
        self.block_length = block_length

    def transform(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        ds = ds.interleave(self.item_to_dataset,
                           cycle_length=self.cycle_length,
                           block_length=self.block_length,
                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return ds


class _CacheFeatureTransform(_FeatureTransform):
    def __init__(self, cache_dir: Text):
        self.cache_dir = cache_dir

    def transform(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        if self.cache_dir is None:
            return ds
        elif self.cache_dir == "":
            log.info("Using memory cache for %s", ds)
            ds = ds.cache()
        else:
            cache_path = os.path.join(self.cache_dir, "cache")
            log.info("Using cache path:[%s] for %s", self.cache_dir, ds)
            tf.io.gfile.makedirs(os.path.dirname(cache_path))
            ds = ds.cache(cache_path)
        return ds


class _AppendFeatureTransform(_FeatureTransform):
    def __init__(self, feature: Feature):
        self.feature = feature

    def transform(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        ds = super()._map_ds(self.feature.get_feature_transform(), ds)
        return ds


class _DatasetFeatureTransform(_FeatureTransform):
    def __init__(self, feature: Feature):
        self.feature = feature

    def transform(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        ds = self.feature.transform_dataset(ds)
        return ds


class _DropFeatureTransform(_FeatureTransform):
    def __init__(self, features: List[Text]):
        self.features = features

    @tf.function
    def drop_feature_fn(self, example: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:  # pragma: no cover
        example = {key: val for key, val in example.items() if key not in self.features}
        return example

    def transform(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        ds = super()._map_ds(self.drop_feature_fn, ds)
        return ds


class _KeepFeatureTransform(_FeatureTransform):
    def __init__(self, features: List[Text]):
        self.features = features

    @tf.function
    def keep_feature_fn(self, example: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:  # pragma: no cover
        example = {key: val for key, val in example.items() if key in self.features}
        return example

    def transform(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        ds = super()._map_ds(self.keep_feature_fn, ds)
        return ds


class BaseDataSet(pp.WithParams):
    class Params(pp.WithParams.Params):
        interleave              = True  # should we interleave the dataset items
        interleave_cycle_length = 4     # should be bigger than the batch_size
        interleave_block_length = 1     #
        cache_dir               = None

    def _construct(self, data_items: List = None, parent=None):
        super()._construct()

        if data_items and not isinstance(data_items, list):
            raise RuntimeError(f"Expected data_items to be a list,"
                               f" but it  is: {type(data_items)}")          # pragma: no cover

        if parent and not isinstance(parent, self.__class__):
            raise RuntimeError(f"Expected parent to be:[{self.__class__}],"
                               f" but it  is: {type(parent)}")              # pragma: no cover

        self._featureTransforms: List[_FeatureTransform] = []
        if parent:
            if parent._featureTransforms:
                self._featureTransforms.extend(parent._featureTransforms)

        self._data_items = data_items
        if self._data_items is None:
            self._data_items = self.construct_data_items()

        if len(self._data_items) < 1:
            raise RuntimeError(f"Dataset is empty. Params: {pprint.pformat(self.params)}")  # pragma: no cover

    @property
    def data_items(self) -> List:
        return self._data_items

    @property
    def params(self) -> Params:
        return self._params

    def construct_data_items(self) -> List:
        """
        Returns the dataset items (dirs, files, item ids).
        :return:
        """
        raise RuntimeError("NotYetImplemented")              # pragma: no cover

    def get_size(self):
        return len(self.data_items)

    def _clone(self, data_items=None, append_transform=None):
        overrided_data_items = self.data_items if data_items is None else data_items
        ds = self.__class__(data_items=overrided_data_items, parent=self, **self.params)
        if append_transform is not None:
            ds._featureTransforms.append(append_transform)
        return ds

    def with_feature(self, feature: Feature):
        return self._clone(append_transform=_AppendFeatureTransform(feature))

    def with_dataset_transform(self, feature: Feature):
        return self._clone(append_transform=_DatasetFeatureTransform(feature))

    def with_cache(self, cache_dir=None):
        cdir = cache_dir if cache_dir is not None else self.params.cache_dir
        return self._clone(append_transform=_CacheFeatureTransform(cdir))

    def with_drop_features(self, features: List[Text]):
        return self._clone(append_transform=_DropFeatureTransform(features))

    def with_keep_features(self, features: List[Text]):
        return self._clone(append_transform=_KeepFeatureTransform(features))

    def split(self, splits=None, train_to_validation_ratio=None, random_seed=7411):
        """
        Splits the dataset.

        Parameters:
            :param splits: list of relative split sizes, i.e. [8,2] would do a 80/20 split
            :param train_to_validation_ratio:  a value of 4. results in a 80/20 split
            :param random_seed: a random seed
        """
        if splits is None:
            ratio = train_to_validation_ratio
            splits = [ratio / (1 + ratio), 1 / (1 + ratio)]
            print("BaseDataSet: using splits:{} from ratio:{}".format(splits, ratio))
        else:
            assert train_to_validation_ratio is None, "Both `splits` an `train_to_validation_ratio` args specified"

        all_items = list(self.data_items)
        random.Random(random_seed).shuffle(all_items)

        assert isinstance(splits, list)
        assert all([split > 0 for split in splits])

        split_ratios = np.array(splits) / np.sum(splits)
        split_points = np.ceil(np.cumsum(split_ratios) * len(all_items)).astype(np.int).tolist()

        print("BaseDataSet: splitting at positions:{}".format(split_points))

        result = []
        for start, end in zip([0] + split_points[:-1], split_points):
            if not (start < end):
                break
            ds = self._clone(data_items=all_items[start:end])
            result.append(ds)

        assert len(result) == len(splits), "Cannot split to {} - too few data:[{}]".format(splits, len(all_items))

        return tuple(result)

    def _create_dataset(self, shuffle=True, interleave=None) -> tf.data.Dataset:
        interleave_ds = self.params.interleave if interleave is None else interleave
        if interleave_ds:
            ds = tf.data.Dataset.from_tensor_slices(self.data_items)
            if shuffle:
                ds = ds.shuffle(buffer_size=len(self.data_items), reshuffle_each_iteration=True)
            ds = _InterleaveTransform(self._item_to_dataset,
                                      cycle_length=self.params.interleave_cycle_length,
                                      block_length=self.params.interleave_block_length).transform(ds)
        else:
            ds = self._item_to_dataset(self.data_items)
            if shuffle:
                ds = ds.shuffle(buffer_size=len(self.data_items))
        return ds

    def _item_to_dataset(self, data_items: Union[List, Any]) -> tf.data.Dataset:
        if not isinstance(data_items, list):
            data_items = [data_items]
        ds = tf.data.Dataset.from_tensor_slices(data_items)
        return ds

    def build_dataset(self, shuffle=True, interleave=None) -> tf.data.Dataset:
        ds = self._create_dataset(shuffle=shuffle, interleave=interleave)
        for featureTransform in self._featureTransforms:
            ds = featureTransform.transform(ds)
        return ds


class Featurizer(pp.WithParams):
    class Params(pp.WithParams.Params):
        pass

    def _construct(self):
        super()._construct()

    @property
    def params(self) -> Params:
        return self._params                                          # pragma: no cover

    def featurize(self, ds: BaseDataSet, train=True) -> BaseDataSet:
        raise RuntimeError("Not implemented")                        # pragma: no cover
