from .backbone import *
import tensorflow as tf


class FeatureExtractor(tf.keras.layers.Layer):

    def __init__(self, input_shape=(96, 96, 3), weights='imagenet'):
        super(FeatureExtractor, self).__init__()
        self.backbone = get_feature_extractor(weights=weights, input_shape=input_shape)
        self.dense = tf.keras.layers.Dense(256)

    def call(self, inputs):
        convolutional_feature_map = self.backbone(inputs)
        reshaped_feature_map = tf.keras.layers.Reshape((36, 1024))(convolutional_feature_map)
        word_embeddings = self.dense(reshaped_feature_map)
        return word_embeddings
