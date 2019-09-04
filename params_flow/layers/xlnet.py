# coding=utf-8
#
# created by kpe on 19.Jul.2019 at 13:06
#

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python import keras

import params_flow as pf


def residual_wrapper(layer):
    class Wrapper(pf.Layer):
        def call(self, inputs, **kwargs):
            return inputs + layer(inputs)
    return Wrapper()


class EmbeddingLookup(pf.Layer):
    class Params(pf.Layer.Params):
        vocab_size  = None
        hidden_size = None
        initializer = None
        use_tpu     = False
        name        = "embedding"
        dtype       = tf.float32

    @property
    def params(self) -> Params: ...

    def build(self, input_shape):
        self.lookup_table = self.add_variable("lookup_table",
                                              shape=(self.params.vocab_size,
                                                     self.params.hidden_size),
                                              initializer=self.create_initializer(self)
                                              )

    def call(self, inputs, **kwargs):
        x = inputs
        if self.params.use_tpu:
            one_hot_idx = tf.one_hot(x, self.params.vocab_size, dtype=self.dtype)
            equation = 'in,nj->ij' if one_hot_idx.shape.ndims == 2 else 'ibn,nj->ibj'
            return tf.einsum(equation, one_hot_idx, self.lookup_table)
        else:
            return tf.nn.embedding_lookup(self.lookup_table, x)


def positional_embedding(pos_seq, inv_freq, bsz=None):
  sinusoid_inp = tf.einsum('i,d->id', pos_seq, inv_freq)
  pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
  pos_emb = pos_emb[:, None, :]

  if bsz is not None:
    pos_emb = tf.tile(pos_emb, [1, bsz, 1])

  return pos_emb


class PositionwiseFFN(pf.Layer):
    class Params(pf.Layer.Params):
        hidden_size = None                    # d_model
        inner_size  = None                    # d_inner
        dropout     = None
        initializer = None
        initializer_range = None
        activation  = "relu"
        name        = "ff"

    @property
    def params(self) -> Params: ...

    def build(self, input_shape):
        self.dense_1 = keras.layers.Dense(self.params.inner_size,
                                          activation=pf.get_activation(self.params.activation),
                                          kernel_initializer=pf.get_initializer(self.params),
                                          name="layer_1")
        self.dense_2 = keras.layers.Dense(self.params.hidden_size,
                                          kernel_initializer=pf.get_initializer(self.params),
                                          name="layer_2")
        self.dropout_1 = keras.layers.Dropout(self.params.dropout, name="drop_1")
        self.dropout_2 = keras.layers.Dropout(self.params.dropout, name="drop_2")
        self.layer_norm = keras.layers.LayerNormalization(norm_axis=-1, name="LayerNorm")

    def call(self, inputs, **kwargs):
        output = inputs
        output = self.dense_1(output)
        output = self.dropout_1(output)
        output = self.dense_2(output)
        output = self.dropout_2(output)
        output = self.layer_norm(output + inputs)
        return output


class HeadProjection(pf.Layer):
    class Params(pf.Layer.Params):
        hidden_size = None               # d_model
        num_heads   = None               # n_head
        head_size   = None               # d_head
        initializer = None               # kernel_initializer
        initializer_range = None
        name        = None

    @property
    def params(self) -> Params: ...

    def build(self, input_shape):
        self.proj_weight = self.add_weight("kernel",
                                           shape=(self.params.hidden_size,
                                                  self.params.num_heads,
                                                  self.params.head_size),
                                           initializer=pf.get_initializer(self.params),
                                           dtype=self.params.dtype)

    def call(self, inputs, **kwargs):
        head = tf.einsum('ibh,hnd->ibnd', inputs, self.proj_weight)
        return head


class PostAttention(pf.Layer):
    class Params(pf.Layer.Params):
        hidden_size = None                 # d_model
        num_heads   = None                 # n_head
        head_size   = None                 # d_head
        dropout     = None
        initializer = None                 # kernel_initializer
        initializer_range = None
        name        = "o"
        residual    = True

    @property
    def params(self) -> Params: ...

    def build(self, input_shape):
        self.proj_o = self.add_weight("kernel",
                                      shape=(self.params.hidden_size,
                                             self.params.num_heads,
                                             self.params.head_size),
                                      initializer=pf.get_initializer(self.params))
        self.dropout = keras.layers.Dropout(self.params.dropout)
        self.layer_norm = pf.LayerNormalization(norm_axis=-1, name="LayerNorm")

    def call(self, inputs, **kwargs):
        x, attn_vec = inputs
        attn_out = tf.einsum('ibnd,hnd->ibh', attn_vec, self.proj_o)
        attn_out = self.dropout(attn_out)
        if self.params.residual:
            attn_out = attn_out + x
        output = self.layer_norm(attn_out)
        return output


class MultiHeadAttentionLayer(pf.Layer):
    class Params(HeadProjection.Params,
                 PostAttention.Params):
        dropout_attention  = None

    @property
    def params(self) -> Params: ...

    def _construct(self, params):
        self.scale = 1 / (self.params.head_size ** 0.5)

    def build(self, input_shape):
        assert isinstance(input_shape, list) and len(input_shape) in [3, 4]
        if len(input_shape) == 3:
            q_shape, k_shape, v_shape = input_shape
            attn_mask_shape = None
        else:
            q_shape, k_shape, v_shape, attn_mask_shape = input_shape
        self.input_spec = [keras.layers.InputSpec(shape=shape) for shape in input_shape]

        self.dropout = keras.layers.Dropout(self.params.dropout_attention)
        self.softmax = keras.layers.Softmax(axis=1)

        self.q_head = HeadProjection.from_params(self.params, name="q")
        self.k_head = HeadProjection.from_params(self.params, name="k")
        self.v_head = HeadProjection.from_params(self.params, name="v")

        self.post_attention = PostAttention.from_params(self.params)

    def _abs_attn_core(self, q_head, k_head, v_head, attn_mask):
        attn_score = tf.einsum("ibnd,jbnd->ijbn", q_head, k_head)
        attn_score *= self.scale
        if attn_mask is not None:
            attn_score = attn_score - 1e30 * attn_mask

        # attention probability
        attn_prob = self.softmax(attn_score)
        attn_prob = self.dropout(attn_prob)

        # attention output
        attn_vec = tf.einsum("ijbn,jbnd->ibnd", attn_prob, v_head)

        return attn_vec

    def call(self, inputs, **kwargs):
        if len(inputs) == 3:
            q, k, v = inputs
            attn_mask = None
        else:
            q, k, v, attn_mask = inputs

        q_head = self.q_head(q)
        k_head = self.k_head(k)
        v_head = self.v_head(v)

        # attention vector
        attn_vec = self._abs_attn_core(q_head, k_head, v_head, attn_mask)

        # post processing
        output = self.post_attention([v, attn_vec])

        return output


class RelMultiHeadAttentionLayer(pf.Layer):
    class Params(HeadProjection.Params,
                 PostAttention.Params):
        dropout_attention  = None
        untie_r            = False     # whether to untie the biases in attention
        num_layers         = None
        use_bfloat16       = False

    @property
    def params(self) -> Params: ...

    def _construct(self, params):
        self.scale = 1 / (self.params.head_size ** 0.5)
        self.tf_float = tf.bfloat16 if self.params.use_bfloat16 else tf.float32

    def build(self, input_shape):
        assert isinstance(input_shape, list) and len(input_shape) in [3, 4]
        if len(input_shape) == 3:
            q_shape, k_shape, v_shape = input_shape
            attn_mask_shape = None
        else:
            q_shape, k_shape, v_shape, attn_mask_shape = input_shape
        self.input_spec = [keras.layers.InputSpec(shape=shape) for shape in input_shape]

        self.dropout = keras.layers.Dropout(self.params.dropout_attention)
        self.softmax = keras.layers.Softmax(axis=1)

        self.q_head = HeadProjection.from_params(self.params, name="q")
        self.k_head = HeadProjection.from_params(self.params, name="k")
        self.v_head = HeadProjection.from_params(self.params, name="v")

        self.k_head_r = HeadProjection.from_params(self.params, name="r")

        self.post_attention = PostAttention.from_params(self.params)

        bias_shape = [self.params.num_heads, self.params.head_size]
        if self.params.untie_r:
            bias_shape = [self.params.num_heads] + bias_shape

        self.r_w_bias = self.add_weight("r_w_bias", shape=bias_shape, initializer=pf.get_initializer(self.params))
        self.r_r_bias = self.add_weight("r_w_bias", shape=bias_shape, initializer=pf.get_initializer(self.params))

        # move this into a seg embedding layer
        seg_id_shape = None
        if seg_id_shape is not None:
            if self.params.untie_r:
                self.r_s_bias = self.add_weight("r_s_bias", shape=bias_shape, initializer=pf.get_initializer(self.params))
            self.seg_embed = self.add_weight("seg_embed", [self.params.num_layers, 2,
                                                           self.params.num_heads, self.params.head_size],
                                             initializer=pf.get_initializer(self.params))

    @staticmethod
    def rel_shift(x, klen=-1):
        """relative shift for the relative attention score."""
        x_size = tf.shape(x)

        x = tf.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
        x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
        x = tf.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
        x = tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])

        return x

    def _rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat, attn_mask):
        # content based attention score
        ac = tf.einsum("ibnd,jbnd->ijbn", q_head, self.r_w_bias, k_head_h)

        # position based attention score
        bd = tf.einsum("ibnd,jbnd->ijbn", q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift(bd, klen=tf.shape(ac)[1])

        # segment based attention score
        if seg_mat is None:
            ef = 0
        else:
            ef = tf.einsum("ibnd,snd->ibns", q_head + self.r_s_bias, seg_embed)
            ef = tf.einsum("ijbs,ibns->ijbn", seg_mat, ef)

        # merge attention scores and perform masking
        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            attn_score = attn_score - 1e30 * attn_mask

        # attention probability
        attn_prob = self.softmax(attn_score)
        attn_prob = self.dropout(attn_prob)

        # attention output
        attn_vec = tf.einsum("ijbn,jbnd->ibnd", attn_prob, v_head_h)

        return attn_vec

    def call(self, inputs, **kwargs):

        # FIXME - unpack input
        h, r, seg_id = None, None, None
        attn_mask = None
        mems = None

        if len(inputs) == 3:
            h, r, seg_id = inputs
        else:
            h, r, seg_id, mems, attn_mask = inputs

        cat = h
        if mems is not None and mems.shape.ndims > 1:
            cat = tf.concat([mems, h], axis=0)

        # content heads
        q_head_h = self.q_head(h)
        k_head_h = self.k_head(cat)
        v_head_h = self.v_head(cat)

        # positional heads
        k_head_r = self.k_head_r(r)

        if seg_id is not None:
            # convert seg_id to one-hot seg_mat
            mem_pad = tf.zeros([mlen, bsz], dtype=tf.int32)
            cat_ids = tf.concat([mem_pad, seg_id], axis=0)

            # 1 indicates not in the same segment [qlen x klen x bsz]
            seg_mat = tf.cast(tf.logical_not(tf.equal(seg_id[:, None],
                                                      cat_ids[None, :])),
                              tf.int32)
            seg_mat = tf.one_hot(seg_mat, depth=2, dtype=self.tf_float)


        # attention vector
        attn_vec = self._rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r,
                                       seg_embed, seg_mat, attn_mask)

        # post processing
        output = self.post_attention([v, attn_vec])

        return output
