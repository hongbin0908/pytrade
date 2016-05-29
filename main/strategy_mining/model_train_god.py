#!/usr/bin/env python
#@author binred@outlook.com

"""
generate the price series of stocks which base on the first day's price
"""
import sys,os,numpy,random
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

from  model_base import *

from model_train_base import TrainRegress
class GodModel:
    def fit(self,x,y):
        pass
    def predict(self, x, sym, datestr, span):
        stock_prices1 = get_stock_data_one_day(sym, datestr)
        stock_prices2 = get_stock_data_span_day(sym, datestr, span)
        return stock_prices2["close"] * 1.0 / stock_prices1["close"] * 10000
class God(TrainRegress):
    def get_model(self):
        return GodModel()
