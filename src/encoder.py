from .utils import *
from .blocks import *
import tensorflow as tf


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, dense_units, attention_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(dense_units, attention_heads)
        self.pw_ffn = PositionWiseFFN(dense_units, dff)
        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        attention, _ = self.attention(x, x, x, mask)
        attention = self.dropout_1(attention, training=training)
        output_1 = self.norm_1(x + attention)
        ffn_out = self.pw_ffn(output_1)
        ffn_out = self.dropout_2(ffn_out, training=training)
        output_2 = self.norm_2(output_1 + ffn_out)
        return output_2


class Encoder(tf.keras.layers.Layer):

    def __init__(
            self, n_encoder_layers, dense_units,
            attention_heads, dff, input_vocab_size,
            max_positional_encoding, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.dense_units = dense_units
        self.n_encoder_layers = n_encoder_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, dense_units)
        self.positional_encoding = positional_encoding(max_positional_encoding, self.dense_units)
        self.encoder_layers = [
            EncoderLayer(
                self.dense_units, attention_heads,
                dff, dropout_rate
            ) for _ in range(self.n_encoder_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        sequence_length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dense_units, dtype=tf.float32))
        x += self.positional_encoding[:, :sequence_length, :]
        x = self.dropout(x, training=training)
        for i in range(self.n_encoder_layers):
            x = self.encoder_layers[i](x, training, mask)
        return x