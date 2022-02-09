import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as kerasBackend
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.utils.data_utils import validate_file
from joblib import Parallel, delayed, parallel
import utils
import os
import numpy as np
import glob
import random
import json
import sys
import math
import pandas as pd
import multiprocessing as mp
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping


# In[2]:


def L1(vectors):
    (featsA, featsB) = vectors
    sum = kerasBackend.sum(kerasBackend.abs(featsA - featsB), axis=1,keepdims=True)
    return kerasBackend.maximum(sum,kerasBackend.epsilon())

def L2(vectors):
    (featsA, featsB) = vectors
    sum = kerasBackend.sqrt(kerasBackend.sum(kerasBackend.square(featsA - featsB), axis=1,keepdims=True))
    return kerasBackend.maximum(sum,kerasBackend.epsilon())

def MSE(vectors):
    (featsA, featsB) = vectors
    mean = kerasBackend.mean(kerasBackend.square(featsA - featsB), axis=1,keepdims=True)
    return kerasBackend.maximum(mean,kerasBackend.epsilon())

def RMSE(vectors):
    (featsA, featsB) = vectors
    mean = kerasBackend.sqrt(kerasBackend.mean(kerasBackend.square(featsA - featsB), axis=1,keepdims=True))
    return kerasBackend.maximum(mean,kerasBackend.epsilon())

def Cosine(vectors):
    (featsA, featsB) = vectors
    featsA = kerasBackend.l2_normalize(featsA)
    featsB = kerasBackend.l2_normalize(featsB)
    similarity = kerasBackend.sum(featsA*featsB,axis=1,keepdims=True)
    return kerasBackend.maximum(similarity,kerasBackend.epsilon())


# In[3]:


def generate_sister_model():
    model = models.Sequential()
    model.add(layers.Conv1D(512,5,activation=layers.LeakyReLU(),input_shape=(2028,1)))
    model.add(layers.Conv1D(256,3,activation=layers.LeakyReLU()))
    model.add(layers.MaxPool1D(pool_size=2,strides=2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv1D(64,13,activation=layers.LeakyReLU()))
    model.add(layers.Conv1D(32,7,activation=layers.LeakyReLU()))
    model.add(layers.MaxPool1D(pool_size=2,strides=2))
    # model.add(layers.Conv1D(128,7,activation='relu'))
    # model.add(layers.Conv1D(64,10,activation='relu'))
    # model.add(layers.MaxPool1D(pool_size=2,strides=2))
    model.add(layers.Flatten())
    
    return model


# In[4]:


def generate_siamese_model():
    sig = layers.Input(shape=(2028,1))
    ref = layers.Input(shape=(2028,1))
    
    model = generate_sister_model()
    sig_model = model(sig)
    ref_model = model(ref)

    # Definir a distância
    distance = layers.Lambda(RMSE)([sig_model, ref_model])

    outputs = layers.Dense(1, activation="sigmoid")(distance)
    siamese_model = models.Model(inputs=[sig, ref], outputs=outputs)
    
    return siamese_model


# # Funções para treinar o model0
# In[6]:


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


# In[7]:

def generate_initial_df(df_path):
    base_df = pd.read_excel(df_path)
    aux_df = pd.DataFrame(columns=["x1",'x2','y'])
    x1 = []
    x2 = []
    y = []
    
    labels = base_df['label'].unique()
    for label in labels:
        print('Starting '+label+' label...')
        a = base_df[base_df['label']==label]
        b = base_df[base_df['label']!=label]
        
        for index, row in a.iterrows():
            # criando par positivo
            x1.append(row['filename'])
            sample = a.sample(n=1)
            x2.append(sample['filename'].values[0])
            y.append(1)

            #criando par negativo
            x1.append(row['filename'])
            sample = b.sample(n=1)
            x2.append(sample['filename'].values[0])
            y.append(0)
        print('Label finished.\n')
            
    aux_df['x1'] = x1
    aux_df['x2'] = x2
    aux_df['y'] = y
    
    return base_df, aux_df


# In[8]:

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, base_df, aux_df,
                 batch_size,
                 input_size=(2028,1),
                 shuffle=True):
        
        self.base_df = base_df
        self.aux_df = aux_df
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.n = len(self.aux_df)
        
    def read_txt_arrays(self, file_list):
        # result = []
        # for file in file_list:
        #     result.append(np.loadtxt(file).tolist())
        # return result
        result = Parallel(n_jobs=8)(delayed(np.loadtxt)(txt) for txt in file_list)
        return result
        
    def on_epoch_end(self):
        self.aux_df = pd.DataFrame(columns=["x1",'x2','y'])
        x1 = []
        x2 = []
        y = []

        labels = self.base_df['label'].unique()
        for label in labels:
            print('Starting '+label+' label...')
            a = self.base_df[self.base_df['label']==label]
            b = self.base_df[self.base_df['label']!=label]

            for index, row in a.iterrows():
                # criando par positivo
                x1.append(row['filename'])
                sample = a.sample(n=1)
                x2.append(sample['filename'].values[0])
                y.append(1)

                #criando par negativo
                x1.append(row['filename'])
                sample = b.sample(n=1)
                x2.append(sample['filename'].values[0])
                y.append(0)
            print('Label finished.\n')

        self.aux_df['x1'] = x1
        self.aux_df['x2'] = x2
        self.aux_df['y'] = y
    
    def __get_data(self, batch):
        x1 = batch['x1'].tolist()
        x2 = batch['x2'].tolist()
        y = batch['y'].tolist()
        
        x1 = self.read_txt_arrays(x1)
        x2 = self.read_txt_arrays(x2)        
        
        x1 = np.array(x1, dtype=float).reshape(self.batch_size,2028,1)
        x2 = np.array(x2, dtype=float).reshape(self.batch_size,2028,1)
        y = np.array(y, dtype=int)
        return ([x1, x2], y)
    
    def __getitem__(self, index):        
        batch = self.aux_df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batch)
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size


# In[9]:

def train_model():
    model = generate_siamese_model()
    model.compile(loss=contrastive_loss, optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy", tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),
                           tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalseNegatives(), 
                           tf.keras.metrics.TruePositives(),tf.keras.metrics.FalsePositives()])
    
    batch_size = 64
    
    train_df, train_aux = generate_initial_df('train.xlsx')
    train_generator = CustomDataGen(train_df, train_aux, batch_size)
    
    validation_df, validation_aux = generate_initial_df('validation.xlsx')
    validation_generator = CustomDataGen(validation_df, validation_aux, batch_size)
    
    es = EarlyStopping(monitor='val_loss',patience=5,min_delta=0.01)
    mc = ModelCheckpoint(filepath='checkpoints/test.h5', monitor='val_loss',
                         verbose=0, save_best_only=True, save_weights_only=False, mode='min', save_freq='epoch')
    
    history = model.fit_generator(train_generator,
                validation_data=validation_generator,
                epochs=1, callbacks=[es,mc],verbose=1, workers=1, use_multiprocessing=False)

    pd.DataFrame.from_dict(history.history).to_excel('history.xlsx', index=False)

train_model()

