#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

import numpy as np
from main.classifier.base_classifier import BaseClassifier

class RandClassifer(BaseClassifier):
    def __init__(self, tp = 0.5):
        self.tp = tp
    def get_name(self):
        "random"
    def fit(self, X, y):
        return self
    def predict_proba(self, X):
        from random import gauss
        y_list = []
        for i in range(len(X)):
            while True:
                value = gauss(self.tp, 0.2)
                if 0 < value < 1:
                    y_list.append(value)
                    break
        y = np.ndarray(shape=(len(X), 2))
        y[:,1] = np.asarray(y_list)
        y[:,0] = 1 - np.asarray(y_list)
        return y
