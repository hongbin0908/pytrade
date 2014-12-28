#!/usr/bin/env python
#@author binred@outlook.com

"""
generate the price series of stocks which base on the first day's price
"""
import sys,os,numpy
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

from model_train_base import TrainRegress
class NothingModel:
    def fit(self,x,y):
        pass
    def predict(self,X):
        return numpy.ones(X.shape[0]) * 10000
class Nothing(TrainRegress):
    def get_model(self):
        return NothingModel()
