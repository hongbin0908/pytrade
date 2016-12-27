#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong


class BaseClassifier:
    def get_name(self):
        pass
    def fit(self, X, y):
        pass
    def predict_proba(self, X):
        pass
