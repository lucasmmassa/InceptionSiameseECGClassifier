import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models
from tensorflow.keras import backend as kerasBackend
from tensorflow.keras.utils import Sequence
from joblib import Parallel, delayed, parallel
import numpy as np
import random
import json
import math
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Input, Dense, Flatten, SpatialDropout2D, Lambda
from tensorflow.keras.utils import plot_model

def inception_module(layer_in, f1, f2, f3):
    # 1x1 conv
    conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
    # 3x3 conv
    conv3 = Conv2D(f2, (1,3), padding='same', activation='relu')(layer_in)
    # 5x5 conv
    conv5 = Conv2D(f3, (1,5), padding='same', activation='relu')(layer_in)
    # 3x3 max pooling
    # pool = MaxPooling2D((2,2), strides=(1,1), padding='same')(layer_in)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5], axis=-1)
    return layer_out

def sister_functional_generation(input_layer):
    inception1 = inception_module(input_layer, 128, 128, 128)
    # inception2 = inception_module(inception1, 128, 128, 128)
    # drop1 = SpatialDropout2D(0.2)(inception2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(inception1)
    inception3 = inception_module(pool1, 64, 64, 64)
    # inception4 = inception_module(inception3, 64, 64, 64)
    # drop2 = SpatialDropout2D(0.2)(inception4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(inception3)
    flat = Flatten()(pool2)
    return flat
    inception5 = inception_module(pool2, 64, 64, 64)
    inception6 = inception_module(inception5, 64, 64, 64)
    pool3 = MaxPooling2D(pool_size=(2, 2))(inception6)

def RMSE(vectors):
    (featsA, featsB) = vectors
    mean = kerasBackend.sqrt(kerasBackend.mean(kerasBackend.square(featsA - featsB), axis=1,keepdims=True))
    return kerasBackend.maximum(mean,kerasBackend.epsilon())

def generate_inception_model():
    sig = layers.Input(shape=(1000, 12, 1))
    ref = layers.Input(shape=(1000, 12, 1))

    sig_model = sister_functional_generation(sig)
    ref_model = sister_functional_generation(ref)

    # Definir a distância
    distance = layers.Lambda(RMSE)([sig_model, ref_model])
    # L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    # distance = L1_layer([sig_model, ref_model])

    outputs = layers.Dense(1, activation="sigmoid")(distance)
    siamese_model = models.Model(inputs=[sig, ref], outputs=outputs)

    return siamese_model

def contrastive_loss(y_true, y_pred, margin=1):
    # y = tf.cast(y, preds.dtype)
    # squaredPreds = kerasBackend.square(preds)
    # squaredMargin = kerasBackend.square(kerasBackend.maximum(margin - preds, 0))
    # loss = kerasBackend.mean((1 - y) * squaredPreds + (y) * squaredMargin)
    # return loss
    y_true = tf.cast(y_true, y_pred.dtype)
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
    return tf.math.reduce_mean(
        (1 - y_true) * square_pred + (y_true) * margin_square
    )

class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, data_x, data_y,
                 batch_size,
                 input_size=(1000, 12, 1),
                 shuffle=True):

        self.data_x = data_x
        self.data_y = data_y
        self.possible_labels = np.unique(data_y)
        self.aux_x1 = None
        self.aux_x2 = None
        self.aux_y = None
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.n = data_y.shape[0]
        self.on_epoch_end()

    def on_epoch_end(self): # shuffled positive and negative pairs
        print('shuffling data...')
        aux_x1 = []
        aux_x2 = []
        aux_y = []

        for label in self.possible_labels:
            indexes = self.data_y == label
            positive_x = self.data_x[indexes]
            negative_x = self.data_x[~indexes]

            for i in range(positive_x.shape[0]):
                # create positive pair
                aux_x1.append(positive_x[i].copy())
                random_idx = random.sample(range(positive_x.shape[0]), 1)[0]
                aux_x2.append(positive_x[random_idx].copy())
                aux_y.append(1.0)

                # create negative pair
                aux_x1.append(positive_x[i].copy())
                random_idx = random.sample(range(negative_x.shape[0]), 1)[0]
                aux_x2.append(negative_x[random_idx].copy())
                aux_y.append(0.0)

        self.aux_x1 = aux_x1
        self.aux_x2 = aux_x2
        self.aux_y = aux_y
        print('finished.\n')

    def __get_data(self, batch_x1, batch_x2, batch_y):
        x1 = batch_x1.reshape(self.batch_size, 1000, 12, 1)
        x2 = batch_x2.reshape(self.batch_size, 1000, 12, 1)
        y = np.array(batch_y, dtype=int)
        return ([x1, x2], y)

    def __getitem__(self, index):
        batch_x1 = np.array(self.aux_x1[index * self.batch_size:(index + 1) * self.batch_size])
        batch_x2 = np.array(self.aux_x2[index * self.batch_size:(index + 1) * self.batch_size])
        batch_y = np.array(self.aux_y[index * self.batch_size:(index + 1) * self.batch_size])
        X, y = self.__get_data(batch_x1, batch_x2, batch_y)
        return X, y

    def __len__(self):
        return self.n // self.batch_size

def train_model():
    model = generate_inception_model()
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                           tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalseNegatives(),
                           tf.keras.metrics.TruePositives(), tf.keras.metrics.FalsePositives()])

    batch_size = 16

    train_x = np.load('x_train.npy')
    train_y = np.load('y_train.npy')
    train_generator = CustomDataGen(train_x, train_y, batch_size)

    val_x = np.load('x_test.npy')
    val_y = np.load('y_test.npy')
    validation_generator = CustomDataGen(val_x, val_y, batch_size)

    es = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01)
    mc = ModelCheckpoint(filepath='test.h5', monitor='val_loss',
                         verbose=0, save_best_only=True, save_weights_only=False, mode='min', save_freq='epoch')

    history = model.fit_generator(train_generator,
                                  validation_data=validation_generator,
                                  epochs=5, callbacks=[es, mc], verbose=1, workers=1, use_multiprocessing=False)

    pd.DataFrame.from_dict(history.history).to_excel('history.xlsx', index=False)

def test():
    model = generate_inception_model()
    # print(model.summary())
    plot_model(model, to_file='model.png')

train_model()
