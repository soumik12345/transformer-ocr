from src.utils import *
from src.layers import *
import tensorflow as tf
from matplotlib import pyplot as plt


# model = FeatureExtractor(weights=None)
# x = tf.random.normal((1, 96, 96, 3))
# y = model(x)
# print(y.shape)

pos_encoding = positional_encoding(36, 256)
plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 256))
plt.ylabel('Position')
plt.colorbar()
plt.title('Positional Encoding Test')
plt.show()
