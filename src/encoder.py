from .blocks import *
import tensorflow as tf


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, n_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attn = MultiHeadAttention(d_model, n_heads)
        self.pffn = point_wise_feed_forward_network(d_model, dff)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = tf.keras.layers.Dropout(rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        attn_output, _ = self.multi_head_attn(x, x, x)
        attn_output = self.dropout_1(attn_output, training=training)
        out_1 = self.layer_norm_1(x + attn_output)
        ffn_output = self.pffn(out_1)
        ffn_output = self.dropout_2(ffn_output, training=training)
        out_2 = self.layer_norm_2(out_1 + ffn_output)
        return out_2
