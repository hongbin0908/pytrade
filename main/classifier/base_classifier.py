#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong
import os
import sys
import pickle
import keras

from abc import ABCMeta, abstractmethod
class BaseClassifier:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    @abstractmethod
    def get_feature_importances(self, feat_names):
        pass

    @abstractmethod
    def save(self, save_path):
        pass

    @abstractmethod
    def load(self, save_path):
        pass

class SklearnClassifier(BaseClassifier):
    __metaclass__ = ABCMeta

    def fit(self, X, y):
        return self.classifier.fit(X, y)

    def save(self, save_path):
        with open(save_path, 'wb') as fout:
            pickle.dump(self.classifier, fout, protocol=-1)

    def load(self, save_path):
        with open(save_path, 'rb') as fin:
            self.classifier = pickle.load(fin)


class KerasClassifier(BaseClassifier):
    __metaclass__ = ABCMeta

    def save(self, save_path):
        self.classifier.save(save_path)

    def load(self, save_path):
        self.classifier = keras.models.load_model(save_path)
