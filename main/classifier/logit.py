#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong
import numpy as np

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

class Logit(BaseClassifier):
    def __init__(self, batch_size = 100, nb_epoch=10, verbose = 1):
        model = Sequential()
        self.classifier = model
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        pass
    def get_name(self):
        return "ccl-logit-%d-%d" % (self.nb_epoch, self.batch_size)
    def fit(self, X, y, X_t, y_t):
        self.classifier.add(Dense(input_dim=X.shape[1], output_dim=1))
        self.classifier.add(Activation('sigmoid'))
        sgd = SGD(lr=0.01)
        opt = Adam(lr=4e-5)
        opt = Adam()
        #opt = RMSprop(lr=4e-3)
        #opt = Adadelta()
        self.classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.classifier.fit(X, y, validation_data=(X_t, y_t), batch_size=self.batch_size, nb_epoch=self.nb_epoch)
    def predict_proba(self, X):
        re = self.classifier.predict_proba(X)
        re = np.hstack([1-re, re])
        return re

    def save(self, save_path):
        self.classifier.save(save_path)

    def load(self, save_path):
        self.classifier = keras.models.load_model(save_path)
