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
    :param i_tag: integer tag denoting the inside of the sequence (could be a vector too).
    Note: b_tag takes preceedence over e_tag, and e_tag over i_tag.
    """
    def _where(cond, tval, fval):
        return tf.cast(cond, tf.int32) * tval + tf.cast(tf.logical_not(cond), tf.int32) * fval
    rstart = _where(tf.less(start, 0), -1, start)
    rend   = _where(tf.less(end, 0), _where(tf.less(start, 0), -1, length), tf.minimum(end, length))
    srange = tf.range(length)
    s_mask = tf.greater(srange, tf.expand_dims(rstart, -1))
    e_mask = tf.less(srange, tf.expand_dims(rend, -1))
    i_seq  = tf.cast(tf.logical_and(s_mask, e_mask), tf.int32) * tf.expand_dims(i_tag, -1)
    b_seq  = tf.cast(tf.equal(srange, tf.expand_dims(rstart, -1)), tf.int32) * b_tag
    e_seq  = tf.cast(tf.equal(srange, tf.expand_dims(rend, -1)), tf.int32) * e_tag
    # tag preceedence begin, end, inside
    result = _where(tf.equal(e_seq, 0), i_seq, e_seq)
    result = _where(tf.equal(b_seq, 0), result, b_seq)
    return result
