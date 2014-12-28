#!/usr/bin/env python
#@author binred@outlook.com

"""
generate the price series of stocks which base on the first day's price
"""
import sys,os,numpy,random
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from model_train_base import TrainRegress
class Gdbc1Model:
    def __init__(self):
        self.model = GradientBoostingClassifier(max_features=0.6, learning_rate=0.05, max_depth=5, n_estimators=300)

    def fit(self,x,y):
        self.model.fit(x,y)
    def predict(self,X):
        return self.model.predict(X)
class Gdbc1(TrainRegress):
    def get_model(self):
        return Gdbc1Model()
