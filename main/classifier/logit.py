#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong
import numpy as np
import pandas as pd

import keras
from keras.layers import Flatten, Activation, Dense, Dropout, K
from keras.layers import LSTM
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Reshape
from keras.models import Sequential
from keras.optimizers import SGD, Adam, Adamax, RMSprop, Adadelta
from keras import initializers
from main.classifier.base_classifier import BaseClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import linear_model
import main.base as base

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops


#from lasagne import layers  
#from lasagne.updates import nesterov_momentum  
#from nolearn.lasagne import NeuralNet 
#from nolearn.lasagne import visualize  
#import lasagne
#import pickle
from sklearn.metrics import confusion_matrix
import matplotlib  
import matplotlib.pyplot as plt  
import matplotlib.cm as cm  
from sklearn import metrics
import numpy as np
#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation
#from keras.layers.embeddings import Embedding
#from keras.layers.recurrent import LSTM
#from keras.optimizers import SGD
#from keras.layers.core import Flatten

# Define functions for initializing variables and standard layers
#For now, this seems superfluous, but in extending the code
#to many more layers, this will keep our code
#read-able
import os, sys
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', "..")
sys.path.append(root)

def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def top_accuracy(y_true, y_pred):
    K.in_top_k(y_pred, y_true)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

class IntervalAcc(Callback):
    def __init__(self, cls, validation_data=(), interval=10):
        super(Callback, self).__init__()
        self.interval = interval
        self.cls = cls
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.cls.predict_proba(self.X_val)
            print(y_pred.shape)
            print(len(y_pred[:,1]), len(self.X_val), len(self.y_val))
            df = pd.DataFrame({"pred": y_pred[:,1], "val": self.y_val})
            df.sort_values(["pred"], ascending=False, inplace=True)
            df1 = df.head(1000)
            score1 = len(df1[df1.val == 1])/len(df1)
            threshold1 = float(df1.tail(1)["pred"].values)
            df2 = df.head(10000)
            score2 = len(df2[df2.val == 1])/len(df2)
            threshold2 = float(df2.tail(1)["pred"].values)
            dfn = df[df.pred >= 0.5]
            if len(dfn) == 0:
                thresholdn = 0.5
                scoren = 0.0
            else:
                scoren = len(dfn[dfn.val == 1])/len(dfn)
                thresholdn = float(dfn.tail(1)["pred"].values)
            print("interval evaluation - epoch: {:d} - threshold: {:.6f} {:.6f} {:.6f} - score: {:.6f} {:.6f} {:.6f}"
                  .format(epoch, threshold1, threshold2,thresholdn,score1,score2,scoren))
class IntervalAuc(Callback):
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_proba(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))

class Logit(BaseClassifier):
    def __init__(self, batch_size = 100, nb_epoch=20, verbose = 1):
        model = Sequential()
        self.classifier = model
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        pass
    def get_name(self):
        return "ccl-logit-%d-%d" % (self.nb_epoch, self.batch_size)
    def fit(self, X, y, X_t, y_t):
        self.classifier.add(Dense(input_dim=X.shape[1], output_dim=32))
        self.classifier.add(Activation('relu'))
        self.classifier.add(Dropout(0.5))
        for i in range(1):
            self.classifier.add(Dense(output_dim=32))
            self.classifier.add(Activation('relu'))
            self.classifier.add(Dropout(0.5))

        self.classifier.add(Dense(output_dim=1))
        self.classifier.add(Activation('sigmoid'))
        sgd = SGD(lr=0.01)
        opt = Adam(lr=4e-5)
        opt = Adam()
        #opt = RMSprop(lr=4e-3)
        #opt = Adadelta()
        from keras.metrics import top_k_categorical_accuracy
        self.classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        #self.classifier.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
        ival = IntervalAcc(cls = self, validation_data=(X_t, y_t), interval=1)
        self.classifier.fit(X, y, validation_data=(X_t, y_t), batch_size=self.batch_size, nb_epoch=self.nb_epoch, callbacks=[ival])
    def predict_proba(self, X):
        re = self.classifier.predict_proba(X)
        re = np.hstack([1-re, re])
        return re

    def save(self, save_path):
        self.classifier.save(save_path)

    def load(self, save_path):
        self.classifier = keras.models.load_model(save_path)
