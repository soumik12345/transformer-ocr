import tensorflow as tf
from .attention import MultiHeadAttention


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h):
        super(Decoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.attention_bot = [
            MultiHeadAttention(model_size, h)
            for _ in range(num_layers)
        ]
        self.attention_bot_norm = [
            tf.keras.layers.BatchNormalization()
            for _ in range(num_layers)
        ]
        self.attention_mid = [
            MultiHeadAttention(model_size, h)
            for _ in range(num_layers)
        ]
        self.attention_mid_norm = [
            tf.keras.layers.BatchNormalization()
            for _ in range(num_layers)
        ]
        self.dense_1 = [
            tf.keras.layers.Dense(512, activation='relu')
            for _ in range(num_layers)
        ]
        self.dense_2 = [
            tf.keras.layers.Dense(model_size)
            for _ in range(num_layers)
        ]
        self.ffn_norm = [
            tf.keras.layers.BatchNormalization()
            for _ in range(num_layers)
        ]
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, encoder_output, pos_encodings):
        embed_out = []
        for i in range(sequence.shape[1]):
            embed = self.embedding(tf.expand_dims(sequence[:, i], axis=1))
            embed_out.append(embed + pos_encodings[i, :])
        embed_out = tf.concat(embed_out, axis=1)
        bot_sub_in = embed_out
        for i in range(self.num_layers):
            bot_sub_out = []
            for j in range(bot_sub_in.shape[1]):
                values = bot_sub_in[:, :j, :]
                attention = self.attention_bot[i](
                    tf.expand_dims(bot_sub_in[:, j, :], axis=1), values)
                bot_sub_out.append(attention)
            bot_sub_out = tf.concat(bot_sub_out, axis=1)
            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.attention_bot_norm[i](bot_sub_out)
            mid_sub_in = bot_sub_out
            mid_sub_out = []
            for j in range(mid_sub_in.shape[1]):
                attention = self.attention_mid[i](
                    tf.expand_dims(mid_sub_in[:, j, :], axis=1), encoder_output)
                mid_sub_out.append(attention)
            mid_sub_out = tf.concat(mid_sub_out, axis=1)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.attention_mid_norm[i](mid_sub_out)
            ffn_in = mid_sub_out
            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)
            bot_sub_in = ffn_out
        logits = self.dense(ffn_out)
        return logits
