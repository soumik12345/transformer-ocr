import tensorflow as tf


def get_feature_extractor(input_shape=(96, 96, 3), weights='imagenet'):
    model = tf.keras.applications.ResNet101(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    layer = model.get_layer('conv4_block23_3_conv').output
    return tf.keras.Model(model.input, layer)