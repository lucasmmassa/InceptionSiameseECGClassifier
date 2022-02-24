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
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import inception_naive
import inception_v1

def RMSE(vectors):
    (featsA, featsB) = vectors
    mean = kerasBackend.sqrt(kerasBackend.mean(kerasBackend.square(featsA - featsB), axis=1,keepdims=True))
    return kerasBackend.maximum(mean,kerasBackend.epsilon())

def generate_inception_model():
    sig = layers.Input(shape=(1000, 12, 1))
    ref = layers.Input(shape=(1000, 12, 1))

    sig_model = inception_v1.sister_functional_generation(sig)
    ref_model = inception_v1.sister_functional_generation(ref)

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

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr

def train_model():
    batch_size = 8
    num_epochs = 30
    steps = int(((9794+1111)/batch_size)*num_epochs)

    initial_lr = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=steps,
        decay_rate=0.01,
        staircase=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    lr_metric = get_lr_metric(optimizer)

    model = generate_inception_model()
    model.compile(loss=contrastive_loss, optimizer=optimizer,
                  metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), lr_metric])

    train_x = np.load('x_train.npy')
    train_y = np.load('y_train.npy')
    train_generator = CustomDataGen(train_x, train_y, batch_size)

    val_x = np.load('x_test.npy')
    val_y = np.load('y_test.npy')
    validation_generator = CustomDataGen(val_x, val_y, batch_size)

    es = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
    mc = ModelCheckpoint(filepath='inception_v1/test4.h5', monitor='val_loss',
                         verbose=0, save_best_only=True, save_weights_only=False, mode='min', save_freq='epoch')

    history = model.fit_generator(train_generator,
                                  validation_data=validation_generator,
                                  epochs=num_epochs, callbacks=[mc], verbose=1, workers=1, use_multiprocessing=False)

    pd.DataFrame.from_dict(history.history).to_excel('inception_v1/history4.xlsx', index=False)

def test():
    model = generate_inception_model()
    # print(model.summary())
    plot_model(model, to_file='model.png')

train_model()
