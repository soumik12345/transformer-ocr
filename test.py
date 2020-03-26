from src.layers import *
import tensorflow as tf


model = FeatureExtractor(weights=None)
x = tf.random.normal((1, 96, 96, 3))
y = model(x)
print(y.shape)
