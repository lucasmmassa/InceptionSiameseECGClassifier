from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras import backend as kerasBackend
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf


import inception_naive
import inception_v1

from scipy import rand
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import json
import math
import os

tf.keras.backend.set_floatx('float16')
DRIVE_PATH = ""

def get_asset(path):
    if DRIVE_PATH == "":
        return path
    return os.path.join(DRIVE_PATH, path)

def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def RMSE(vectors):
    (featsA, featsB) = vectors
    mean = kerasBackend.sqrt(kerasBackend.mean(kerasBackend.square(featsA - featsB), axis=1,keepdims=True))
    return kerasBackend.maximum(mean,kerasBackend.epsilon())

def generate_inception_model():
    sig = layers.Input(shape=(1000, 12, 1))
    ref = layers.Input(shape=(1000, 12, 1))
    model_input = layers.Input(shape=(1000, 12, 1))

    backbone = Model(model_input, inception_v1.sister_functional_generation(model_input))
    sig_model = backbone(sig)
    ref_model = backbone(ref)
    # backbone.summary()

    # Definir a distância
    distance = layers.Lambda(RMSE)([sig_model, ref_model])
    # L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    # distance = L1_layer([sig_model, ref_model])

    # normalization = tf.keras.layers.BatchNormalization()(distance)
    outputs = layers.Dense(1, activation="sigmoid")(distance)
    siamese_model = models.Model(inputs=[sig, ref], outputs=outputs)

    return siamese_model

def contrastive_loss(y_true, y_pred, margin=1):
    y_true = tf.cast(y_true, y_pred.dtype)
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
    return tf.math.reduce_mean(
        (1 - y_true) * square_pred + (y_true) * margin_square
    )

class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self, X, y,
                 batch_size,
                 input_size=(1000, 12, 1)):
        self.X = X
        self.y = y
        self.possible_labels = np.unique(y)
        
        self.reference_entry = None
        self.inference_entry = None
        self.labels = None

        self.batch_index = 0
        self.batch_size = batch_size
        self.input_size = input_size
        self.dataset_size = y.shape[0]
        
    def augment(self, entry):
        # possible_increment = [float(i/10) for i in range(-10, 10)]
        # increment = random.choice(possible_increment)
        # entry += increment

        # possible_shift = [i for i in range(-30, 30)]
        # shift = random.choice(possible_shift)
        # entry = np.roll(entry, shift, axis=0)
        return entry

    def shuffle_data(self): # shuffled positive and negative pairs
        print('shuffling data...')
        reference_entry = []
        inference_entry = []
        labels = []

        indexes = np.array([i for i in range(self.dataset_size)])
        np.random.shuffle(indexes)

        similar_samples = {}
        different_samples = {}

        for l in self.possible_labels:
            similar_samples[l] = self.X[self.y == l]
            different_samples[l] = self.X[self.y != l]
        
        for index in tqdm(indexes):
            current_label = self.y[index]

            reference_entry.append(self.augment(self.X[index].copy()))
            inference_entry.append(random.choice(similar_samples[current_label]).copy())
            labels.append(1.0)

            reference_entry.append(self.augment(self.X[index].copy()))
            inference_entry.append(random.choice(different_samples[current_label]).copy())                
            labels.append(0.0)

        self.reference_entry = np.array_split(reference_entry, self.__len__() + 1)
        self.inference_entry = np.array_split(inference_entry, self.__len__() + 1)
        self.labels = np.array_split(labels, self.__len__() + 1)
        self.batch_index = 0
        print('finished.\n')

    def prepare_batch(self, index):
        batch_x1 = self.reference_entry[index]
        batch_x2 = self.inference_entry[index]
        batch_y = self.labels[index].astype(int)

        return ([batch_x1, batch_x2], batch_y)

    def __getitem__(self, index):
        X, y = self.prepare_batch(self.batch_index)
        self.batch_index += 1

        return X, y

    def __len__(self):
        return self.dataset_size // self.batch_size

class ReshuffleDataCallback(Callback):
    def __init__(self, train_generator, validation_generator):
        super().__init__()
        self.train_generator = train_generator
        self.validation_generator = validation_generator

    def on_epoch_begin(self, epoch, logs=None):
        self.train_generator.shuffle_data()
        self.validation_generator.shuffle_data()

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr

def train_model():
    batch_size = 8
    num_epochs = 30

    train_x = np.load(get_asset('x_train.npy'))
    train_y = np.load(get_asset('y_train.npy'))
    train_generator = CustomDataGen(train_x, train_y, batch_size)
    train_generator.shuffle_data()

    val_x = np.load(get_asset('x_test.npy'))
    val_y = np.load(get_asset('y_test.npy'))
    validation_generator = CustomDataGen(val_x, val_y, batch_size)
    validation_generator.shuffle_data()

    steps = int(((train_y.shape[0] + val_y.shape[0])/batch_size) * num_epochs)

    initial_lr = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=steps,
        decay_rate=0.3,
        staircase=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    lr_metric = get_lr_metric(optimizer)

    model = generate_inception_model()
    model.compile(loss=contrastive_loss, optimizer=optimizer,
                  metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), lr_metric])

    reshuffle_data_callback = ReshuffleDataCallback(train_generator, validation_generator)

    mc = ModelCheckpoint(filepath='inception_v1/test4.h5', monitor='val_loss',
                         verbose=0, save_best_only=True, save_weights_only=False, mode='min', save_freq='epoch')

    model.summary()

    history = model.fit(train_generator, validation_data=validation_generator, epochs=num_epochs, 
                        callbacks=[mc, reshuffle_data_callback])

    pd.DataFrame.from_dict(history.history).to_excel('inception_v1/history4.xlsx', index=False)

def test():
    model = generate_inception_model()
    # print(model.summary())
    plot_model(model, to_file='model.png')

train_model()