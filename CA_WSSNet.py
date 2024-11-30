from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Softmax
from tensorflow.keras.models import Model
import tensorflow as tf


def CA_WSSNet(input_shape):
    inputs = Input(shape=input_shape)
    query = Conv2D(4, 3, padding='same', activation='relu', use_bias=False, data_format='channels_last')(inputs)
    key = Conv2D(4, 3, padding='same', activation='relu', use_bias=False, data_format='channels_last')(inputs)
    value = Conv2D(4, 3, padding='same', activation='relu', use_bias=False, data_format='channels_last')(inputs)
    q = tf.reshape(query, (-1, tf.shape(query)[1], tf.shape(query)[2] * 4))
    q = tf.transpose(q, perm=[0, 2, 1])
    k = tf.reshape(key, (-1, tf.shape(key)[1], tf.shape(key)[2] * 4))
    attention = Softmax(axis=-1)(tf.matmul(q, k) / tf.sqrt(float(40)))
    v = tf.reshape(value, (-1, tf.shape(value)[1], tf.shape(value)[2] * 4))
    ca_output = tf.reshape(tf.matmul(v, tf.transpose(attention, perm=[0, 2, 1])), (-1, tf.shape(value)[1], tf.shape(value)[2], 4))
    conv1_output = Conv2D(4, (3, 3), padding='same', activation='relu', use_bias=False, data_format='channels_last')(ca_output)
    flat_output = Flatten()(conv1_output)
    dense1_output = Dense(128, activation='relu')(flat_output)
    dense2_output = Dense(40, activation='sigmoid')(dense1_output)
    return Model(inputs=inputs, outputs=dense2_output)