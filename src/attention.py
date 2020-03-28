import tensorflow as tf


class MultiHeadAttention(tf.keras.Model):

    def __init__(self, model_size, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.key_size = model_size // self.n_heads
        self.query_size = model_size // self.n_heads
        self.value_size = model_size // self.n_heads
        self.wq = [
            tf.keras.layers.Dense(self.query_size)
            for _ in range(self.n_heads)
        ]
        self.wk = [
            tf.keras.layers.Dense(self.key_size)
            for _ in range(self.n_heads)
        ]
        self.wv = [
            tf.keras.layers.Dense(self.value_size)
            for _ in range(self.n_heads)
        ]
        self.wo = tf.keras.layers.Dense(model_size)

    def call(self, decoder_out, encoder_out):
        heads = []
        for i in range(self.n_heads):
            score = tf.matmul(self.wq[i](decoder_out), self.wk[i](encoder_out), transpose_b=True)
            score = score / tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32))
            alignment = tf.nn.softmax(score, axis=2)
            head = tf.matmul(alignment, self.wv[i](encoder_out))
            heads.append(head)
        heads = tf.concat(heads, axis=2)
        heads = self.wo(heads)
        return heads
