#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong
import numpy as np

import keras
from keras.layers import Flatten, Activation, Dense, Dropout
from keras.layers import LSTM
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Reshape
from keras.models import Sequential
from keras.optimizers import SGD, Adam
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
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = name)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

class cnn(BaseClassifier):
    def __init__(self, batch_size = 100, nb_epoch=10, num_filt_1 = 16, num_filt_2 = 14, num_fc_1 = 40, verbose = 1):
        model = Sequential()
        self.classifier = model
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.num_filt_1 = num_filt_1
        self.num_filt_2 = num_filt_2
        self.num_fc_1 = num_fc_1
        self.verbose = verbose
        pass
    def transfer_shape(self,X):
        return np.reshape(X, (X.shape[0], X.shape[1],1,1))
    def get_name(self):
        return "ccl-cnn-%d-%d-%d-%d" % (self.nb_epoch, self.num_filt_1, self.num_filt_2, self.batch_size)
    def fit(self, X, y, X_t, y_t):
        X = self.transfer_shape(X)
        X_t = self.transfer_shape(X_t)
        y = keras.utils.to_categorical(y, 2)
        y_t = keras.utils.to_categorical(y_t, 2)
        print(X.shape)
        if len(X_t) % 2==0:
            X_test, X_val = np.split(X_t, 2)
            y_test, y_val = np.split(y_t, 2)
        else:
            X_test, X_val = np.split(X_t[:-1,:], 2)
            y_test, y_val = np.split(y_t[:-1], 2)
        assert(len(X_test) == len(y_test))

        """Hyperparameters"""
        num_filt_1 = self.num_filt_1     #Number of filters in first conv layer
        num_filt_2 = self.num_filt_2      #Number of filters in second conv layer
        num_fc_1 = self.num_fc_1      #Number of neurons in hully connected layer

        initializer = initializers.glorot_uniform(seed=123)
        self.classifier.add(Conv2D(filters=num_filt_1, kernel_size=[5,1], padding='same',
                                   kernel_initializer=initializer,
                                   bias_initializer=initializers.zeros(),
                                   input_shape=X.shape[1:]))
        self.classifier.add(Activation('relu'))

        self.classifier.add(Conv2D(filters=num_filt_2, kernel_size=[4,1],
                                   kernel_initializer=initializer,
                                   bias_initializer=initializers.zeros(),
                                   padding='same'))

        #self.classifier.add(BatchNormalization())
        self.classifier.add(Activation('relu'))
        self.classifier.add(Flatten())
        self.classifier.add(Dense(num_fc_1, kernel_initializer=initializer,bias_initializer=initializer))
        self.classifier.add(Activation('relu'))
        self.classifier.add(Dropout(0.2, seed=123))

        self.classifier.add(Dense(2, kernel_initializer=initializer, bias_initializer=initializers.Constant(0.1)))

        self.classifier.add(Activation('softmax'))
        #self.classifier.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate),metrics=['accuracy'])
        self.classifier.compile(loss='binary_crossentropy', optimizer='sgd',metrics=['accuracy'])
        self.classifier.fit(X, y, verbose=self.verbose,validation_data=(X_t, y_t), batch_size=self.batch_size, nb_epoch=self.nb_epoch, shuffle=False)
    def predict_proba(self, X):
        X = self.transfer_shape(X)
        re = self.classifier.predict_proba(X, verbose=0)
        return re
def d2tod3(fro, window):
    row = fro.shape[0]
    feat_num = fro.shape[1]

    d1 = row - window +1
    d2 = window
    d3 = feat_num

    print(d1,d2,d3)
    to = np.zeros(d1*d2*d3).reshape(d1,d2,d3)
    for i in range(len(fro)-window + 1):
        to[i] = fro[i:i+window]
    return to
class ccl2(BaseClassifier):
    def __init__(self, batch_size = 32, nb_epoch=10):
        model = Sequential()
        self.classifier = model
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        pass
    def get_name(self):
        return "ccl-2-%d" % (self.nb_epoch)

    def transfer_shape(self,X):
        return d2tod3(X, window=2)
        return np.reshape(X, (X.shape[0], 1, X.shape[1]))

    def fit(self, X, y, X_t, y_t):

        X = self.transfer_shape(X)
        X_t = self.transfer_shape(X_t)
        y = y[2-1:]
        y_t = y_t[2-1:]
        #self.classifier.add(Dense(500, input_shape=( X.shape[1],)))
        self.classifier.add(LSTM(input_shape=(2, X.shape[2]),  output_dim =8,
                                 return_sequences = True, kernel_initializer=initializers.glorot_normal(123) ))
        self.classifier.add(Flatten())
        #self.classifier.add(Activation('linear'))
        self.classifier.add(Activation('relu'))
        self.classifier.add(Dense( output_dim=8, kernel_initializer=initializers.glorot_normal(123)))
        #self.classifier.add(Activation('linear'))
        self.classifier.add(Activation('relu'))
        self.classifier.add(Dropout(0.3, seed=123))
        self.classifier.add(Dense(output_dim=8))
        self.classifier.add(Activation('tanh'))
        self.classifier.add(Dense(output_dim=1, kernel_initializer=initializers.glorot_normal(123)))
        self.classifier.add(Activation('sigmoid'))
        sgd = SGD(lr=0.01)
        opt = Adam(lr=2e-5)

        self.classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.classifier.fit(X, y, validation_data=(X_t, y_t), batch_size=self.batch_size, nb_epoch=self.nb_epoch)
    def predict_proba(self, X):
        X = self.transfer_shape(X)
        re = self.classifier.predict_proba(X)
        re = np.hstack([1-re, re])
        return re
class ccl(BaseClassifier):
    def __init__(self, batch_size = 32, nb_epoch=10):
        model = Sequential()
        self.classifier = model
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        pass
    def get_name(self):
        return "ccl-%d" % (self.nb_epoch)

    def transfer_shape(self,X):
        return d2tod3(X, window=2)
        return np.reshape(X, (X.shape[0], 1, X.shape[1]))

    def fit(self, X, y, X_t, y_t):
        X = self.transfer_shape(X)
        X_t = self.transfer_shape(X_t)
        y = y[2-1:]
        y_t = y_t[2-1:]
        #self.classifier.add(Dense(500, input_shape=( X.shape[1],)))
        self.classifier.add(LSTM(input_shape=(2, X.shape[2]),  output_dim =8,
                                 return_sequences = True ))
        self.classifier.add(Flatten())
        #self.classifier.add(Activation('linear'))
        self.classifier.add(Activation('relu'))
        self.classifier.add(Dense( output_dim=4))
        #self.classifier.add(Activation('linear'))
        self.classifier.add(Activation('relu'))
        self.classifier.add(Dropout(0.3, seed=7))
        self.classifier.add(Dense(output_dim=4))
        self.classifier.add(Activation('relu'))
        self.classifier.add(Dense(output_dim=1))
        self.classifier.add(Activation('sigmoid'))
        sgd = SGD(lr=0.01)
        self.classifier.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self.classifier.fit(X, y, validation_data=(X_t, y_t), batch_size=self.batch_size, nb_epoch=self.nb_epoch)
    def predict_proba(self, X):
        X = self.transfer_shape(X)
        re = self.classifier.predict_proba(X)
        re = np.hstack([1-re, re])
        return re

class MyLogisticRegressClassifier(BaseClassifier):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    def __init__(self, C = 0.01, max_iter=2000):
        #self.classifier = linear_model.LogisticRegression(C=C, max_iter=2000, verbose=1, n_jobs = 30, tol=1e-6, solver='sag')
        #self.name = "lr-%f" % C
        self.classifier = linear_model.LogisticRegression(C=C, max_iter=2000, verbose=1, n_jobs = 30,  penalty='l2', tol = 1e-5)
        self.name = "lr-%f-%d" % (C,max_iter)
    def get_name(self):

        return self.name

    def fit(self, X, y):
        return self.classifier.fit(X, y)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_feature_importances(self, feat_names):
        ipts = dict(zip(feat_names, self.classifier.coef_[0]))
        return ipts

    def verify_predict(self, df):
        feat_names = base.get_feat_names(df)
        ipts = self.get_feature_importances(feat_names)
        s = 0
        for each in ipts:
            if int(df[each]) == 1 :
                s += ipts[each] * 1
        import math
        return 1 / (1 + math.exp(-1 * (s + self.classifier.intercept_)))

class MySGDClassifier(BaseClassifier):
    def __init__(self, n_iter=100):
        self.classifier = linear_model.SGDClassifier(alpha=0.01, loss='log', n_iter=n_iter, verbose=1)
        #self.classifier = linear_model.SGDClassifier(loss='modified_huber', n_iter=n_iter, verbose=1)
        self.name = "sgd-%d"%(n_iter) 
    def get_name(self):

        return self.name

    def fit(self, X, y):
        return self.classifier.fit(X, y)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_feature_importances(self, feat_names):
        ipts = dict(zip(feat_names, self.classifier.coef_[0]))
        return ipts
    def verify_predict(self, df):
        feat_names = base.get_feat_names(df)
        ipts = self.get_feature_importances(feat_names)
        
        s = None
        for each in ipts:
            tmp = df[each]*ipts[each]
            if s is None:
                s = tmp
            else:
                s += tmp
        return 1 / (1 + np.exp(-1 * (s + self.classifier.intercept_)))

class MyRandomForestClassifier(BaseClassifier):
    def __init__(self, verbose=1, n_estimators = 2000, max_depth=8, min_samples_leaf=10000,
                 n_jobs=40):
        self.classifier = RandomForestClassifier( **{'verbose': verbose,
                                                     'n_estimators': n_estimators,
                                                     'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf,
                                                      'n_jobs': n_jobs})
        self.name = "rf_n{n}_md{md}_ms{ms}".format(
            **{"n": n_estimators, "md": max_depth, "ms": min_samples_leaf}
        )

    def get_name(self):
        return self.name

    def fit(self, X, y, X_t, y_t):
        return self.classifier.fit(X, y)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_feature_importances(self, feat_names):
        ipts = dict(zip(feat_names, self.classifier.feature_importances_))
        return ipts


class MyRfClassifier(BaseClassifier):
    def __init__(self, n_estimators, max_depth, min_samples_leaf):
        self.classifier = RandomForestClassifier(**{'verbose':1, 'n_estimators': n_estimators,
                                                    'max_depth':max_depth,'min_samples_leaf':min_samples_leaf,
                                                    'n_jobs':40})
        self.name = "rf_n{n}_md{md}_ms{ms}".format(
            **{"n": n_estimators, "md": max_depth, "ms": min_samples_leaf}
        )
    def get_name(self):
        return self.name

    def fit(self, X, y, X_t, y_t):
        return self.classifier.fit(X, y)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_feature_importances(self, feat_names):
        ipts = dict(zip(feat_names, self.classifier.feature_importances_))
        return ipts
class RFCv1n2000md6msl100(MyRfClassifier):
    def __init__(self):
        n_estimators = 2000
        max_depth = 6
        min_samples_leaf = 100
        MyRfClassifier.__init__(self, n_estimators, max_depth, min_samples_leaf)
class RFCv1n2000md3msl100(MyRfClassifier):
    def __init__(self):
        n_estimators = 2000
        max_depth = 3
        min_samples_leaf = 100
        MyRfClassifier.__init__(self, n_estimators, max_depth, min_samples_leaf)
class RFCv1n2000md2msl100(MyRfClassifier):
    def __init__(self):
        n_estimators = 2000
        max_depth = 2
        min_samples_leaf = 100
        MyRfClassifier.__init__(self, n_estimators, max_depth, min_samples_leaf)
class RFCv1n200md2msl100(MyRfClassifier):
    def __init__(self):
        n_estimators = 200
        max_depth = 2
        min_samples_leaf = 100
        MyRfClassifier.__init__(self, n_estimators, max_depth, min_samples_leaf)
class RFCv1n2000md6msl10000(MyRfClassifier):
    def __init__(self):
        n_estimators = 2000
        max_depth = 6
        min_samples_leaf = 10000
        MyRfClassifier.__init__(self, n_estimators, max_depth, min_samples_leaf)


class MyGradientBoostingClassifier(BaseClassifier):
    def __init__(self, verbose=1, n_estimators=5, max_depth=6, min_samples_leaf=100):
        self.classifier = GradientBoostingClassifier( **{'verbose': verbose,
                                                     'n_estimators': n_estimators,
                                                     'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf
                                                     })
        self.name = "gb_n{n}_md{md}_ms{ms}".format(
            **{"n": n_estimators, "md": max_depth, "ms": min_samples_leaf}
        )

    def get_name(self):
        return self.name

    def fit(self, X, y):
        return self.classifier.fit(X, y)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_feature_importances(self, feat_names):
        ipts = dict(zip(feat_names, self.classifier.feature_importances_))
        return ipts

class GBCv1n600md3lr001(MyGradientBoostingClassifier):
    def __init__(self):
        n_estimators = 600
        max_depth = 3
        learning_rate = 0.01
        self.classifier = GradientBoostingClassifier(**{'verbose': 1, 'n_estimators': n_estimators,
                                                        'max_depth': max_depth, 'learning_rate': learning_rate})
        self.name = "gb_n{n}_md{md}_lr{lr}".format(
            **{"n": n_estimators, "md": max_depth, "lr": learning_rate}
        )

class MyBayesClassifier(BaseClassifier):
    def __init__(self):
        self.classifier = GaussianNB()
        self.name = "bayes"

    def get_name(self):
        return self.name

    def fit(self, X, y):
        return self.classifier.fit(X, y)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_feature_importances(self, feat_names):
        return []
