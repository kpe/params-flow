# coding=utf-8
#
# created by kpe on 29.Nov.2019 at 15:58
#

from __future__ import absolute_import, division, print_function

import tensorflow as tf


def iob_seq(length, start, end, b_tag, e_tag, i_tag):
    """
    Creates an IOB sequence from inclusive start end indices and length.
    :param length: length of the sequence
    :param start: start index. Tagged with b_tag.
    :param end:   end index (inclusive). Tagged with e_tag.
    :param b_tag: integer tag denoting the beginning of the sequence
    :param e_tag: integer tag denoting the end of the sequence
    :param i_tag: integer tag denoting the inside of the sequence
    Note: b_tag takes preceedence over e_tag, and e_tag over i_tag.
    """
    rstart = tf.where(tf.less(start, 0), tf.where(tf.less(end, 0), -1, -1), start)
    rend   = tf.where(tf.less(end, 0), tf.where(tf.less(start, 0), -1, length), tf.minimum(end, length))
    srange = tf.range(length)
    s_mask = tf.greater(srange, tf.expand_dims(rstart, -1))
    e_mask = tf.less(srange, tf.expand_dims(rend, -1))
    i_seq  = tf.cast(tf.logical_and(s_mask, e_mask), tf.int32) * i_tag
    b_seq  = tf.cast(tf.equal(srange, tf.expand_dims(rstart, -1)), tf.int32) * b_tag
    e_seq  = tf.cast(tf.equal(srange, tf.expand_dims(rend, -1)), tf.int32) * e_tag
    # tag preceedence begin, end, inside
    result = tf.where(tf.equal(e_seq, 0), i_seq, e_seq)
    result = tf.where(tf.equal(b_seq, 0), result, b_seq)
    return result
