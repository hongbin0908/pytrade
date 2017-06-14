#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong
import numpy as np

import keras
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Adam, Adadelta
import os, sys

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', "..")
sys.path.append(root)

from main.classifier.base_classifier import BaseClassifier
from main.classifier.interval_acc import IntervalAcc

class Logit(BaseClassifier):
    def __init__(self, dim = 64, hs = 3, batch_size = 100, nb_epoch=30, verbose = 1):
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.dim = dim
        self.hs = hs
        pass
    def get_name(self):
        return "ccl-logit2-%d-%d-%d-%d" % (self.nb_epoch, self.batch_size, self.dim, self.hs)
    def fit(self, X, y, df_test, score):
        import numpy as np
        np.random.seed(608317)
        model = Sequential()
        self.classifier = model
        self.classifier.add(Dense(input_dim=X.shape[1], output_dim=self.dim,
                                  kernel_initializer=keras.initializers.glorot_uniform(seed=570255),
                                  bias_initializer=keras.initializers.constant(0.0)))
        self.classifier.add(Activation('relu'))
        self.classifier.add(Dropout(0.5, seed=969458))
        for i in range(self.hs):
            self.classifier.add(Dense(output_dim=self.dim, kernel_initializer=keras.initializers.glorot_normal(seed=846635)))
            self.classifier.add(Activation('relu'))
            self.classifier.add(Dropout(0.5 ,seed=14306))

        self.classifier.add(Dense(output_dim=1, kernel_initializer=keras.initializers.glorot_uniform(seed=447630),
                                  bias_initializer=keras.initializers.constant(0.0)))
        self.classifier.add(Activation('sigmoid'))
        #opt = SGD(lr=0.01)
        opt = Adam(lr=4e-5)
        opt = Adadelta(lr=4e-5)
        self.classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        ival = IntervalAcc(cls = self, validation_data=(df_test, score), interval=1)
        self.classifier.fit(X, y, shuffle=False, batch_size=self.batch_size, nb_epoch=self.nb_epoch, callbacks=[ival])
    def predict_proba(self, X):
        re = self.classifier.predict_proba(X, verbose=0)
        re = np.hstack([1-re, re])
        return re

    def save(self, save_path):
        self.classifier.save(save_path)

    def load(self, save_path):
        self.classifier = keras.models.load_model(save_path)
