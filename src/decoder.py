from .blocks import *
import tensorflow as tf


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, n_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.multi_head_attn_1 = MultiHeadAttention(d_model, n_heads)
        self.multi_head_attn_2 = MultiHeadAttention(d_model, n_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = tf.keras.layers.Dropout(rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate)
        self.dropout_3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training):
        attn1, attn_weights_block1 = self.multi_head_attn_1(x, x, x)
        attn1 = self.dropout_1(attn1, training=training)
        out1 = self.layer_norm_1(attn1 + x)
        attn2, attn_weights_block2 = self.multi_head_attn_2(enc_output, enc_output, out1)
        attn2 = self.dropout_2(attn2, training=training)
        out2 = self.layer_norm_2(attn2 + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout_3(ffn_output, training=training)
        out3 = self.layer_norm_3(ffn_output + out2)
        return out3, attn_weights_block1, attn_weights_block2
