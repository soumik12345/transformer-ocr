import numpy as np
from src.utils import *
from src.encoder import *
from src.layers import *
import tensorflow as tf
from matplotlib import pyplot as plt


# Test Feature Extractor
# model = FeatureExtractor(weights=None)
# x = tf.random.normal((1, 96, 96, 3))
# y = model(x)
# print(y.shape)


# Test Positional Encoding
# pos_encoding = positional_encoding(36, 256)
# print(pos_encoding.shape)
# plt.pcolormesh(pos_encoding, cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 256))
# plt.ylabel('Position')
# plt.colorbar()
# plt.title('Positional Encoding Test')
# plt.show()


feature_extractor = get_feature_extractor(weights=None)
encoder = Encoder(36, 256, 4, 2)

input_image = tf.random.normal((1, 96, 96, 3))
features = FeatureExtractor(weights=None)(input_image)
pos_encoding = positional_encoding(36, 256)
print(features.shape, pos_encoding.shape)
encoder_out = encoder(features, pos_encoding)
print(encoder_out.shape)