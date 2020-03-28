import tensorflow as tf
from .attention import MultiHeadAttention


class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, model_size, n_layers, n_heads):
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.attention = [
            MultiHeadAttention(model_size, n_heads)
            for _ in range(self.n_layers)
        ]
        self.attention_norm = [
            tf.keras.layers.BatchNormalization()
            for _ in range(self.n_layers)
        ]
        self.dense_1 = [
            tf.keras.layers.Dense(512, activation='relu')
            for _ in range(self.n_layers)
        ]
        self.dense_2 = [
            tf.keras.layers.Dense(model_size)
            for _ in range(self.n_layers)
        ]
        self.ffn_norm = [
            tf.keras.layers.BatchNormalization()
            for _ in range(self.n_layers)
        ]

    def call(self, sequence, pos_encodings):
        sub_in = []
        for i in range(sequence.shape[1]):
            embed = self.embedding(tf.expand_dims(sequence[:, i], axis=1))
            sub_in.append(embed + pos_encodings[i, :])
        sub_in = tf.concat(sub_in, axis=1)
        for i in range(self.num_layers):
            sub_out = []
            for j in range(sub_in.shape[1]):
                attention = self.attention[i](
                    tf.expand_dims(sub_in[:, j, :], axis=1), sub_in)
                sub_out.append(attention)
            sub_out = tf.concat(sub_out, axis=1)
            sub_out = sub_in + sub_out
            sub_out = self.attention_norm[i](sub_out)
            ffn_in = sub_out
            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_in + ffn_out
            ffn_out = self.ffn_norm[i](ffn_out)
            sub_in = ffn_out
        return ffn_out
