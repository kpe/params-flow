# coding=utf-8
#
# created by kpe on 29.Nov.2019 at 16:02
#

from __future__ import absolute_import, division, print_function

import unittest
import tensorflow as tf

from params_flow.utils import iob_seq


def novec_iob_seq(length, start, end, b_tag, e_tag, i_tag):
    """
    Creates an IOB sequence.
    :param length: length of the sequence
    """
    rstart = tf.where(tf.less(start, 0), tf.where(tf.less(end,   0), -1, -1),     start)
    rend   = tf.where(tf.less(end,   0), tf.where(tf.less(start, 0), -1, length), tf.minimum(end, length))
    b_seq = tf.ones(tf.cast(tf.logical_and(tf.less(-1, rstart), tf.less(rstart, length)), tf.int32), tf.int32) * b_tag
    i_seq = tf.ones(tf.maximum(0, rend - rstart - 1), tf.int32) * i_tag
    e_seq = tf.ones(tf.cast(tf.logical_and(tf.less(rstart, rend), tf.less(rend, length)), tf.int32), tf.int32) * e_tag
    pre_seq  = tf.fill((tf.maximum(0, start),), 0)
    post_seq = tf.fill((length - tf.minimum(tf.maximum(rend+1, 0), length),), 0)
    iobs  = tf.concat([pre_seq, b_seq, i_seq, e_seq, post_seq], axis=0)
    return iobs


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.enable_eager_execution()

    def test_iob_seq_vec_tag(self):
        self.assertEqual(iob_seq(6, [2, 1], [4, 4], 1, 2, [7, 9]).numpy().tolist(),
                         [[0, 0, 1, 7, 2, 0],
                          [0, 1, 9, 9, 2, 0]])

    def test_iob_seq_vec(self):
        self.assertEqual(iob_seq(5, [2, 1], [3, 4], 1, 2, 7).numpy().tolist(),
                         [[0, 0, 1, 2, 0],
                          [0, 1, 7, 7, 2]])

        iobs = iob_seq(5,
                       [-1, -1,-1,-1, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5,-1,-1,-1,-2,-2,-2],
                       [-1,  0, 1, 2, 2, 2, 3, 4, 4,-1,-1, 4, 5, 6, 6, 5, 4, 3,-1,-3], 1, 2, 7)
        expected = [
            [0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [7, 2, 0, 0, 0],
            [7, 7, 2, 0, 0],
            [1, 7, 2, 0, 0],
            [0, 1, 2, 0, 0],
            [0, 1, 7, 2, 0],
            [0, 0, 1, 7, 2],
            [0, 0, 0, 1, 2],
            [0, 0, 0, 1, 7],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7],
            [7, 7, 7, 7, 2],
            [7, 7, 7, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0], ]

        tf.assert_equal(expected, iobs)

    def test_iob_seq(self):
        self.assertEqual(iob_seq(5, -1, -1, 1, 2, 7).numpy().tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(iob_seq(5, -1,  0, 1, 2, 7).numpy().tolist(), [2, 0, 0, 0, 0])
        self.assertEqual(iob_seq(5, -1,  1, 1, 2, 7).numpy().tolist(), [7, 2, 0, 0, 0])
        self.assertEqual(iob_seq(5, -1,  2, 1, 2, 7).numpy().tolist(), [7, 7, 2, 0, 0])
        self.assertEqual(iob_seq(5,  0,  2, 1, 2, 7).numpy().tolist(), [1, 7, 2, 0, 0])
        self.assertEqual(iob_seq(5,  1,  2, 1, 2, 7).numpy().tolist(), [0, 1, 2, 0, 0])
        self.assertEqual(iob_seq(5,  1,  3, 1, 2, 7).numpy().tolist(), [0, 1, 7, 2, 0])
        self.assertEqual(iob_seq(5,  2,  4, 1, 2, 7).numpy().tolist(), [0, 0, 1, 7, 2])
        self.assertEqual(iob_seq(5,  3,  4, 1, 2, 7).numpy().tolist(), [0, 0, 0, 1, 2])
        self.assertEqual(iob_seq(5,  3, -1, 1, 2, 7).numpy().tolist(), [0, 0, 0, 1, 7])
        self.assertEqual(iob_seq(5,  4, -1, 1, 2, 7).numpy().tolist(), [0, 0, 0, 0, 1])
        self.assertEqual(iob_seq(5,  4,  4, 1, 2, 7).numpy().tolist(), [0, 0, 0, 0, 1])
        self.assertEqual(iob_seq(5,  5,  5, 1, 2, 7).numpy().tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(iob_seq(5,  5,  6, 1, 2, 7).numpy().tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(iob_seq(5, -1,  6, 1, 2, 7).numpy().tolist(), [7, 7, 7, 7, 7])
        self.assertEqual(iob_seq(5, -1,  5, 1, 2, 7).numpy().tolist(), [7, 7, 7, 7, 7])
        self.assertEqual(iob_seq(5, -1,  4, 1, 2, 7).numpy().tolist(), [7, 7, 7, 7, 2])
        self.assertEqual(iob_seq(5, -2,  3, 1, 2, 7).numpy().tolist(), [7, 7, 7, 2, 0])
        self.assertEqual(iob_seq(5, -2, -1, 1, 2, 7).numpy().tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(iob_seq(5, -2, -3, 1, 2, 7).numpy().tolist(), [0, 0, 0, 0, 0])

    def test_iob_seq_ragged(self):
        self.assertEqual(iob_seq(5,
                                 tf.ragged.constant([-1, -2]),
                                 tf.ragged.constant([-1,  3]), 1, 2, 7).numpy().tolist(),
                         [[0, 0, 0, 0, 0],
                          [7, 7, 7, 2, 0]])

    def test_iob_seq_vec_tag_ragged(self):
        self.assertEqual(iob_seq(6,
                                 tf.ragged.constant([[2, 1], [1]]),
                                 tf.ragged.constant([[4, 4], [3]]), 1, 2,
                                 tf.ragged.constant([[7, 9], [9]])).to_list(),
                         [[[0, 0, 1, 7, 2, 0],
                           [0, 1, 9, 9, 2, 0]],
                          [[0, 1, 9, 2, 0, 0]]])
