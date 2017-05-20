#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong
import numpy as np

import keras
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Adam
import os, sys

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', "..")
sys.path.append(root)

from main.classifier.base_classifier import BaseClassifier
from main.classifier.interval_acc import IntervalAcc

class Logit2(BaseClassifier):
    def __init__(self, batch_size = 100, nb_epoch=1, verbose = 1):
        model = Sequential()
        self.classifier = model
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        pass
    def get_name(self):
        return "ccl-logit-%d-%d" % (self.nb_epoch, self.batch_size)
    def fit(self, X, y, df_test, score):
        self.classifier.add(Dense(input_dim=X.shape[1], output_dim=64))
        self.classifier.add(Activation('relu'))
        self.classifier.add(Dropout(0.5))
        for i in range(2):
            self.classifier.add(Dense(output_dim=64))
            self.classifier.add(Activation('relu'))
            self.classifier.add(Dropout(0.5))

        self.classifier.add(Dense(output_dim=1))
        self.classifier.add(Activation('sigmoid'))
        sgd = SGD(lr=0.01)
        opt = Adam(lr=4e-5)
        self.classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        ival = IntervalAcc(cls = self, validation_data=(df_test, score), interval=1)
        self.classifier.fit(X, y, batch_size=self.batch_size, nb_epoch=self.nb_epoch, callbacks=[ival])
    def predict_proba(self, X):
        re = self.classifier.predict_proba(X, verbose=0)
        re = np.hstack([1-re, re])
        return re

    def save(self, save_path):
        self.classifier.save(save_path)

    def load(self, save_path):
        self.classifier = keras.models.load_model(save_path)