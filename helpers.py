from keras.callbacks import TensorBoard
from keras.layers import Activation, BatchNormalization, Conv2D, Lambda
from keras import backend as K
import numpy as np
import cv2
import tensorflow as tf

def Conv2D_BN(X, filters, kernel_size, strides, padding, name, activation="relu", use_bn=True):
    """Wrapper for creating Conv2D + BatchNormalization (optional) + Activation (optional)

    :param X: input tensor [None, width, height, channels]
    :param filters: number of filters
    :param kernel_size: kernel size, same format as in Keras Conv2D
    :param strides: stride, same format as in Keras Conv2D
    :param padding: padding, same format as in Keras Conv2D
    :param name: layer name
    :param activation: activation function (in the format acceptable to Activation class in Keras), or None for no
        activation function (== linear, f(x) = x). Default: "relu"
    :param use_bn: whether to use batch normalization
    :return: output tensor
    """
    X = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name)(X)
    if use_bn:
        bn_name = name + "_BN"
        X = BatchNormalization(scale=False, name=bn_name)(X)
    if activation is not None:
        ac_name = name + "_" + activation
        X = Activation("relu", name=ac_name)(X)
    return X

def shortcut_summation(X, X_shortcut, scale, name):
    """Summation of shortcut and inception tensors

    :param X: post-inception tensor, [None, width, height, channels]
    :param X_shortcut: shortcut tensor, [None, width, height, channels]
    :param scale: scale factor. Suggested values in article are between 0.1 and 0.3
    :param name: layer name
    :return: output tensor
    """
    X = Lambda(lambda inputs, scale: inputs[0] * scale + inputs[1],
               output_shape=K.int_shape(X)[1:],
               arguments={"scale": scale},
               name=name)([X, X_shortcut])
    return X

def contrastive_loss(y_true, y_pred, margin=0.5):
    """Loss function for minimizing distance between similar images and maximizing between dissimilar
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    :param y_true: ground truth labels, 0 if images similar and 1 if dissimilar
    :param y_pred: predicted labels
    :param margin: maximum distance between dissimilar images at which they are still considered for loss calculations
    """
    left_output = y_pred[0::2, :]
    right_output = y_pred[1::2, :]
    dist = tf.reduce_sum(tf.square(left_output - right_output), 1)
    dist_sqrt = tf.sqrt(dist)
    L_s = dist
    L_d = tf.square(tf.maximum(0.0, margin - dist_sqrt))
    loss = y_true * L_d + (1 - y_true) * L_s
    return loss