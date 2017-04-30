#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong
import numpy as np
import pandas as pd

import keras
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import os, sys

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', "..")
sys.path.append(root)

from main.classifier.base_classifier import BaseClassifier
import main.base as base


class IntervalAcc(Callback):
    def __init__(self, cls, validation_data=(), interval=10):
        super(Callback, self).__init__()
        self.interval = interval
        self.cls = cls
        self.df_test_valid, self.score = validation_data
        #self.df_test = self.df_test_valid.sample(frac=0.5, random_state=200)
        self.df_test_valid = self.df_test_valid.sort_values("date", ascending=True)
        self.df_test = self.df_test_valid.head(int(len(self.df_test_valid)/2))
        self.df_valid = self.df_test_valid.drop(self.df_test.index)
        assert(len(self.df_valid) + len(self.df_test) == len(self.df_test_valid))

        self.npFeatTest, self.npLabelTest = base.extract_feat_label(self.df_test, self.score, drop=True)
        self.npFeatVal, self.npLabelVal = base.extract_feat_label(self.df_valid, self.score, drop=True)

    def cal_accuracy(self, npFeat, npLabel):
        y_pred = self.cls.predict_proba(npFeat)
        df = pd.DataFrame({"pred": y_pred[:,1], "val": npLabel})
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
        return ((threshold1, threshold2, thresholdn),(score1,score2,scoren))

    def cal_accuracy2(self, npFeat, npLabel, thresholds):
        y_pred = self.cls.predict_proba(npFeat)
        df = pd.DataFrame({"pred": y_pred[:,1], "val": npLabel})
        df.sort_values(["pred"], ascending=False, inplace=True)
        df1 = df[df.pred >= thresholds[0]]
        score1 = len(df1[df1.val == 1])/len(df1)
        df2 = df[df.pred >= thresholds[1]]
        score2 = len(df2[df2.val == 1])/len(df2)
        dfn = df[df.pred >= thresholds[2]]
        if len(dfn) == 0:
            scoren = 0.0
        else:
            scoren = len(dfn[dfn.val == 1])/len(dfn)
        return (score1,score2,scoren)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            print("interval evaluation - epoch: %d" % epoch)
            (thresholds, scores) = self.cal_accuracy(self.npFeatTest, self.npLabelTest)
            print()
            print("TEST: ", end='')
            for i in range(len(thresholds)):
                print("score: %.3f(%.3f)" % (scores[i], thresholds[i]), end=" ")
            print()
            print("VALD: ", end='')
            scores = self.cal_accuracy2(self.npFeatVal, self.npLabelVal, thresholds)
            for i in range(len(thresholds)):
                print("score: %.3f(%.3f)" % (scores[i], thresholds[i]), end=" ")
            print()

class IntervalAuc(Callback):
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_proba(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print()
            print("epoch: {:d} - score: {:.6f}".format(epoch, score))

class Logit2(BaseClassifier):
    def __init__(self, batch_size = 100, nb_epoch=20, verbose = 1):
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
