import tensorflow as tf
import numpy as np

from tensorflow import keras
from scipy.io import loadmat


# def bulid_model(X, Y):
#     # value = [batch, in_width, channels]
#     # filter = [filter_width, in_channels, out_channels]
#     conv_1_out = tf.compat.v1.nn.conv1d(value=X, filters=[40, 1, 32], padding='VALID')
#
#     # ksize is the conv_size of each dimension [batch, in_width, in_channels]
#     max_pool_1_out = tf.compat.v1.nn.max_pool1d(conv_1_out, ksize=[1, 4, 1])
#     drop_out_1 = tf.compat.v1.nn.dropout(max_pool_1_out, rate=0.1)
#     activation_1_out = tf.nn.relu(drop_out_1)
#     batch_norm_1 = tf.nn.batch_normalization(activation_1_out)
#
#     conv_2_out = tf.compat.v1.nn.conv1d(value=batch_norm_1, filters=[40, 32, 32], padding='VALID')
#     max_pool_2_out = tf.compat.v1.nn.max_pool1d(conv_2_out, ksize=[1, 4, 32])
#     drop_out_2 = tf.compat.v1.nn.dropout(max_pool_2_out, rate=0.1)
#     activation_2_out = tf.nn.relu(drop_out_2)
#     batch_norm_2 = tf.nn.batch_normalization(activation_2_out)
#
#     lstm_1 = tf.keras.layers.LSTM(128, activation=tf.nn.tanh)
#     lstm_1_out = lstm_1.get_initial_state(batch_norm_2)
#     lstm_2 = tf.compat.v1.keras.layers.LSTM(128, activation=tf.nn.tanh)
#     lstm_2_out = lstm_2.get_initial_state(lstm_1_out)
#
#     flatten = tf.keras.layers.Flatten(lstm_2_out)


def build_model(X):
    model = keras.Sequential()
    '''
    filters:
    Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
    When using this layer as the first layer in a model,
    provide an input_shape argument (tuple of integers or None, e.g. (10, 128)
    for sequences of 10 vectors of 128-dimensional vectors, or (None, 128) for variable-length sequences of 128-dimensional vectors.
    kernel_size:
    An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
    '''
    model.add(
        keras.layers.Conv1D(filters=32, kernel_size=40, activation=keras.activations.relu))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(pool_size=4))
    model.add(keras.layers.Dropout(rate=0.1))

    model.add(
        keras.layers.Conv1D(filters=32, kernel_size=40, activation=keras.activations.relu))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(pool_size=4))
    model.add(keras.layers.Dropout(rate=0.1))

    model.add(keras.layers.LSTM(128, activation=keras.activations.tanh, return_sequences=True))
    model.add(keras.layers.LSTM(128, activation=keras.activations.tanh))

    # model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(3, activation=keras.activations.softmax))
    model.build(X.shape)
    model.summary()

    return model


if __name__ == '__main__':

    data_channels = 1

    data = loadmat('./dataset/row_ppg.mat').get('row_ppg')
    label = loadmat('./dataset/labels.mat').get('labels')

    label = label - 1

    X_train = data[:5000, ]
    X_test = data[5000:, ]

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    Y_train = label[:5000, ]
    Y_test = label[5000:, ]

    # Y_train = keras.backend.one_hot(Y_train_orig, 3)
    # Y_test = keras.backend.one_hot(Y_test_orig, 3)
    # #
    # X = tf.placeholder(dtype=tf.float32, shape=[None, 1000, 1], name='X')
    # Y = tf.placeholder(dtype=tf.float32, shape=[1], name='Y')

    model = build_model(X_train)
    model.compile(optimizer='RMSprop',
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=10, batch_size=512, verbose=1, validation_data=(X_test, Y_test))

