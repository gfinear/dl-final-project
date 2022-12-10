import math
import numpy as np
import tensorflow as tf


class AttentionMatrix(tf.keras.layers.Layer):

    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask

    def call(self, inputs):
        K, Q = inputs
        window_size_queries = Q.get_shape()[1] 
        window_size_keys    = K.get_shape()[1]  

        mask_vals = np.triu(np.ones((window_size_queries, window_size_keys)) * np.NINF, k=1)
        mask = tf.convert_to_tensor(value=mask_vals, dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])

        product = tf.matmul(Q, K, transpose_b=True)

        scaled_product = product / tf.math.sqrt(tf.cast(window_size_keys, tf.float32))

        if self.use_mask == True:
            scaled_product += atten_mask
        
        attention = tf.nn.softmax(scaled_product)
        return attention


class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention
        shape = (input_size, output_size)
        self.K = self.add_weight(name = "K", shape=shape)
        self.V = self.add_weight(name = "V", shape=shape)
        self.Q = self.add_weight(name = "Q", shape=shape)
        self.attn_mtx = AttentionMatrix(use_mask = is_self_attention)


    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        K = tf.tensordot(inputs_for_keys, self.K, 1)
        V = tf.tensordot(inputs_for_values, self.V, 1)
        Q = tf.tensordot(inputs_for_queries, self.Q, 1)

        attention = self.attn_mtx((K,Q))

        values = attention @ V

        return values


class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, emb_sz, use_mask, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        return None


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz, MultiHeadedAttention=False, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.ff_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dense(emb_sz)])

        self.self_atten         = AttentionHead(emb_sz, emb_sz, True)  if not MultiHeadedAttention else MultiHeadedAttention(emb_sz, True)
        self.self_context_atten = AttentionHead(emb_sz, emb_sz, False) if not MultiHeadedAttention else MultiHeadedAttention(emb_sz, False)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    @tf.function
    def call(self, inputs, context_sequence):
        outputs1 = self.self_atten(inputs, inputs, inputs)
        outputs1 = inputs + outputs1
        outputs1 = self.layer_norm(outputs1)

        outputs2 = self.self_context_atten(context_sequence, context_sequence, outputs1)
        outputs2 = outputs1 + outputs2
        outputs2 = self.layer_norm(outputs2)

        outputs3 = self.ff_layer(outputs2)
        outputs3 = outputs2 + outputs3
        outputs3 = self.layer_norm(outputs3)

        return outputs3


def positional_encoding(length, depth):
    ## REFERENCE: https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]   
    depths = np.arange(depth)[np.newaxis, :]/depth 
    angle_rates = 1 / (10000**depths)             
    angle_rads = positions * angle_rates         
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
    return tf.cast(pos_encoding, tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=embed_size)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x
