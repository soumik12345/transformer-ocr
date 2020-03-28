import tensorflow as tf


def get_feature_extractor(input_shape=(96, 96, 3), weights='imagenet'):
    model = tf.keras.applications.ResNet101(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    layer = model.get_layer('conv4_block23_3_conv').output
    return tf.keras.Model(model.input, layer)


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
