from src.backbone import *
import tensorflow as tf


model = get_feature_extractor(weights=None)
x = tf.random.normal((1, 96, 96, 3))
y = model(x)
print(y.shape)