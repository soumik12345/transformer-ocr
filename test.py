import numpy as np
from src.utils import *
from src.blocks import *
from src.layers import *
import tensorflow as tf
from src.encoder import *
from src.decoder import *
from matplotlib import pyplot as plt


# Test Feature Extractor
# model = FeatureExtractor(weights=None)
# x = tf.random.normal((1, 96, 96, 3))
# y = model(x)
# print(y.shape)

# Test Positional Encoding
# pos_encoding = positional_encoding(36, 256)
# plt.pcolormesh(pos_encoding[0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 256))
# plt.ylabel('Position')
# plt.colorbar()
# plt.title('Positional Encoding Test')
# plt.show()


# Test Scaled Dot Product Attention
# def test_scaled_dot_product_attention(q, k, v):
#     temp_out, temp_attn = scaled_dot_product_attention(q, k, v)
#     print('Attention weights are:')
#     print(temp_attn)
#     print('Output is:')
#     print(temp_out)
#
#
# np.set_printoptions(suppress=True)
# temp_k = tf.constant([[10, 0, 0],
#                       [0, 10, 0],
#                       [0, 0, 10],
#                       [0, 0, 10]], dtype=tf.float32)
# temp_v = tf.constant([[1, 0],
#                       [10, 0],
#                       [100, 5],
#                       [1000, 6]], dtype=tf.float32)
# temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)
# test_scaled_dot_product_attention(temp_q, temp_k, temp_v)
# temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)
# test_scaled_dot_product_attention(temp_q, temp_k, temp_v)
# temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)
# test_scaled_dot_product_attention(temp_q, temp_k, temp_v)
# temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)
# test_scaled_dot_product_attention(temp_q, temp_k, temp_v)


# Test Multi Head Attention
# temp_mha = MultiHeadAttention(d_model=512, n_heads=8)
# y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
# out, attn = temp_mha(y, k=y, q=y)
# print(out.shape, attn.shape)


# Point-wise Feed Forward Network
# sample_ffn = point_wise_feed_forward_network(512, 2048)
# print(sample_ffn(tf.random.uniform((64, 50, 512))).shape)


# Encoder Layer
sample_encoder_layer = EncoderLayer(512, 8, 2048)
sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False)
print(sample_encoder_layer_output.shape)


# Decoder Layer
sample_decoder_layer = DecoderLayer(512, 8, 2048)
sample_decoder_layer_output, _, _ = sample_decoder_layer(
    tf.random.uniform((64, 50, 512)), sample_encoder_layer_output, False)
print(sample_decoder_layer_output.shape)
