#-*-encoding:gbk-*-
import sys
import os
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from price_judgement import *
from two_crow_builder import *
from three_inside_pattern import *
import numpy
from sklearn import tree

class base_model:
    #定义基本属性
    name = "base_model"
    #特征生成方法列表，其中，每一个特征的参数形式均为(prices_list, index,feature_result_list),其中，index表示该特征生成的下标，生成的结果会存放在feature_result_list下
    feature_builder_list = []
    sample_judgement = None
    model_predictor = None
    samples = []
    classes = []
    def __init__(self, feature_builder_list_input, sample_judgement_input, model_predictor_input):
        self.feature_builder_list = feature_builder_list_input
        self.sample_judgement = sample_judgement_input
        self.model_predictor = model_predictor_input

    def build_sample(self, open_price_ist, high_price_list, low_price_list, close_price_list, adjust_close_list, volume_list, timewindow):
        for s in range(timewindow, len(open_price_list)-timewindow):
            prices = close_price_list[s-timewindow:s+timewindow]
            price_judge_high = high_price_list[s-timewindow:s]
            price_judge_low = low_price_list[s-timewindow:s]
            price_judge_open = open_price_list[s-timewindow:s]
            price_judge_close = close_price_list[s-timewindow:s]
            adjust_close = adjust_close_list[s-timewindow:s]
            volume = volume_list[s-timewindow:s]

            result = self.sample_judgement.judge(prices, 0.05)
            if result == None:
                continue
            result_list = []
            for m in range(0, len(self.feature_builder_list)):
                result_list.append(0)

            for mindex, m in enumerate(feature_builder_list):
                m.feature_build(price_judge_open, price_judge_high, price_judge_low, price_judge_close, adjust_close, volume, mindex, result_list)
            self.samples.append(result_list)
            self.classes.append(result)
            tmp_str = str(result)
            for s in result_list:
                tmp_str = tmp_str + "\t" + str(s)
            print tmp_str

    def model_process(self):
        model_predictor.fit(numpy.array(self.samples), numpy.array(self.classes))
        predict_value = model_predictor.predict_proba(numpy.array(samples))
        precision, recall, threshold = roc_curve(numpy.array(classes), predict_value[:,0])
        print precision
        print recall
        print threshold
        area = auc(recall, precision)
        print "auc = %.4f" %(area)

    def result_predict(self, open_price_list, high_price_list, low_price_list, close_price_list, adjust_close_list, volume_list, timewindow):
        samples = []
        
        open_price = open_price_list[-timewindow-1: -1]
        high_price = high_price_list[-timewindow-1: -1]
        low_price = low_price_list[-timewindow-1:-1]
        close_price = close_price_list[-timewindow - 1 : -1]
        adjust_price = adjust_close_list[-timewindow - 1 : -1]
        volume = volume_list[-timewindow - 1 : -1]
        for index, s in enumerate(self.feature_builder_list):
            samples.append(m.feature_build(price_judge_open, price_judge_high, price_judge_low, price_judge_close, adjust_close, volume, mindex, result_list))
        predict_value = self.model_predictor.predict(samples)
        return predict_value
 
    def result_predictprob(self, open_price_list, high_price_list, low_price_list, close_price_list, adjust_close_list, volume_list, timewindow):
        samples = []
        
        open_price = open_price_list[-timewindow-1: -1]
        high_price = high_price_list[-timewindow-1: -1]
        low_price = low_price_list[-timewindow-1:-1]
        close_price = close_price_list[-timewindow - 1 : -1]
        adjust_price = adjust_close_list[-timewindow - 1 : -1]
        volume = volume_list[-timewindow - 1 : -1]
        for index, s in enumerate(self.feature_builder_list):
            samples.append(m.feature_build(price_judge_open, price_judge_high, price_judge_low, price_judge_close, adjust_close, volume, mindex, result_list))
        predict_value = self.model_predictor.predict_prob(samples)
        return predict_value

if __name__ == "__main__":
    judger = prices_judgement()
    two_crow = twocrow_builder()
    three_inside = three_inside_up_builder()
    feature_builder_list = []
    feature_builder_list.append(two_crow)
    feature_builder_list.append(three_inside)

    model_predictor = tree.DecisionTreeClassifier()
    model = base_model(feature_builder_list, judger, model_predictor)
    open_prices = []
    high_prices = []
    low_prices = []
    close_prices = []
    adjust_close = []
    volume = []
    file_list = []
    for s in file_list:
        load_data(file_list, open_prices, high_prices, low_prices, close_prices, adjust_close, volume)
        model.build_sample(open_prices, high_prices, low_prices, close_prices, adjust_close, volume, 14)
   
    model.model_process()
#    input_test = numpy.array([-100, 100])
#    print model.result_predict(input_test)
