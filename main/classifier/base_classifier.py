#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong


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
