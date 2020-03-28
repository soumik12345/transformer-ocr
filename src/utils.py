import numpy as np
import tensorflow as tf


def positional_encoding(max_length, model_size):

    def compute_pe(position):
        encoding = np.zeros((1, model_size))
        for i in range(model_size):
            if i % 2 == 0:
                encoding[:, i] = np.sin(position / 10000 ** (i / model_size))
            else:
                encoding[:, i] = np.cos(position / 10000 ** ((i - 1) / model_size))
        return encoding

    embedding = [compute_pe(i) for i in range(max_length)]
    embedding = np.concatenate(embedding, axis=0)
    embedding = tf.constant(embedding, dtype=tf.float32)
    return embedding
