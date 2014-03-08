#-*-encoding:gbk-*-

import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

import numpy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn import linear_model
from sklearn import tree
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import neural_network
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor 

from model_traing_features import *

class base_model:
    #特征生成方法列表，其中，每一个特征的参数形式均为(prices_list, index,feature_result_list),其中，index表示该特征生成的下标，生成的结果会存放在feature_result_list下
    feature_builder_list = []
    sample_judgement = None
    model_predictor = None
    samples = []
    classes = []
    int_num = 0 # give the visual presition of the precess
    def __init__(self, feature_builder_list_input, sample_judgement_input, model_predictor_input):
        self.feature_builder_list = feature_builder_list_input
        self.sample_judgement = sample_judgement_input
        self.model_predictor = model_predictor_input

    def build_sample(self, open_price_list, high_price_list, low_price_list, close_price_list, adjust_close_list, volume_list, timewindow):
        """
        timewindow: 
        """
        self.int_num = self.int_num + 1
        samples = []
        if len(open_price_list) < timewindow * 2:
            return
        for mindex, m in enumerate(self.feature_builder_list):
            # tolist is to ignore the memery copy of numy list
            samples.append(m.feature_build(open_price_list, high_price_list, low_price_list, close_price_list, adjust_close_list, volume_list, mindex, timewindow).tolist())
        tmp_array = numpy.nan_to_num(numpy.column_stack(samples))
        tmp_prices = self.sample_judgement.judge(adjust_close_list, 0.05, 7)
        #判断是否是无效的
        for s in range(0, tmp_array.shape[0]):
            if (numpy.any(tmp_array[s]!=0 )) and (tmp_prices[s] != -2 ):
                self.samples.append(tmp_array[s])
                self.classes.append(tmp_prices[s])
        if self.int_num%10 == 0:
           print self.int_num
    def post_process(self):
        tmp_samples = numpy.array(self.samples)
        self.samples = tmp_samples
        tmp_prices = numpy.array(self.classes)
        self.classes = tmp_prices
        print self.samples.shape

    def model_process(self):
        if len(self.samples) == 0:
            return
        train_size = len(self.samples) * 0.6
        print "DEBUG: train size: %d, predict size: %d" % (train_size, len(self.samples) - train_size)
        self.model_predictor.fit(self.samples[0:train_size], self.classes[0:train_size])
        predict_value = self.model_predictor.predict(self.samples[train_size:])
        r2_score = metrics.r2_score(self.classes[train_size:], predict_value)
        print r2_score

    def result_predict(self, open_price_list, high_price_list, low_price_list, close_price_list, adjust_close_list, volume_list, timewindow):
        samples = []
        if len(open_price_list) < timewindow:
            print "ERROR: the size of input is less than %d" % (timewindow)
            return -1      
        open_price = open_price_list[-timewindow-1:]
        high_price = high_price_list[-timewindow-1:]
        low_price = low_price_list[-timewindow-1:]
        close_price = close_price_list[-timewindow - 1 :]
        adjust_price = adjust_close_list[-timewindow - 1 :]
        volume = volume_list[-timewindow - 1 :]
        try:
            for index, s in enumerate(self.feature_builder_list):
                samples.append(s.feature_build(open_price_list, high_price_list, low_price_list, close_price_list, adjust_close_list, volume_list, index, timewindow)[-1])
        except Exception,e:
            sys.stderr.write("ERROR exception: %s\n" %(e))
            return [0]
        tmp_sample = numpy.nan_to_num(numpy.array(samples))
        if numpy.all(tmp_sample == 0):
            print "ERROR: the sample data are all zero!"
            return [0]
        predict_value = self.model_predictor.predict(tmp_sample)
        if predict_value > 1.12:
            return [predict_value]
        else:
            return [0]

    def dump_model(self):
        # TODO
        pass        

def main():
    judger = prices_judgement()
    feature_builder_list = build_features()
    model_predictor = GradientBoostingRegressor()
    model = base_model(feature_builder_list, judger, model_predictor)
    file_list = get_file_list()
    for s in file_list:
        open_prices, high_prices, low_prices, close_prices, adjust_close_prices,volume = get_stock_data(s)
        if len(open_prices) < 7:
            continue
        model.build_sample(numpy.array(open_prices[:-7]),
                           numpy.array(high_prices[:-7]),
                           numpy.array(low_prices[:-7]),
                           numpy.array(close_prices[:-7]),
                           numpy.array(adjust_close_prices[:-7]),
                           numpy.array(volume[:-7]), 7)
    model.post_process()
    model.model_process()
    num  = 0
    for s in file_list:
        open_prices, high_prices, low_prices, close_prices, adjust_close_prices,volume = get_stock_data(s)
        if len(open_prices) < 7:
            continue
        result = model.result_predict(numpy.array(open_prices), numpy.array(high_prices), numpy.array(low_prices), numpy.array(close_prices),numpy.array(adjust_close), numpy.array(volume), 7)
        if result == None:
            print "ERROR: model.result_predict of %s error" % s
            continue
        if result[0] >1.12 and result[0] > 0:
            num = num + 1
            print "%s\t%.4f" %(s, result[0])
    print num
def get_stock_data(filename):
    """
    input filename : the path of stock daily data
    """
    open_prices = []
    high_prices = []
    low_prices = []
    close_prices = []
    adjust_close_prices = []
    volume = []
    load_data(filename, open_prices, high_prices, low_prices, close_prices, adjust_close_prices, volume)
    open_prices.reverse()
    high_prices.reverse()
    low_prices.reverse()
    close_prices.reverse()
    adjust_close_prices.reverse()
    volume.reverse()
    return open_prices, high_prices, low_prices, close_prices, adjust_close_prices, volume
    
def get_file_list():
    """hongbin0908@126.com
    a help function to load test data.
    """
    file_list = []
    for f in os.listdir('/home/work/workplace/stock_data/'):
        if f != None and not f.endswith(".csv"):
            continue
        file_list.append(os.path.join("/home/work/workplace/stock_data/", f))
         
    return file_list
if __name__ == "__main__":
    main()
