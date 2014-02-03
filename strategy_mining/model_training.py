#-*-encoding:gbk-*-
import sys
import os
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from price_judgement import *
from two_crow_builder import *
import numpy

class base_model:
    #定义基本属性
    name = "base_model"
    #特征生成方法列表，其中，每一个特征的参数形式均为(prices_list, index,feature_result_list),其中，index表示该特征生成的下标，生成的结果会存放在feature_result_list下
    feature_builder_list = []
    sample_judgement = None
    def __init__(self, feature_builder_list_input, sample_judgement_input):
        self.feature_builder_list = feature_builder_list_input
        self.sample_judgement = sample_judgement_input
   
    def model_process(self, open_price_list, high_price_list, low_price_list, close_price_list, timewindow):
        samples = []
        classes = []
        for s in range(timewindow, len(open_price_list)-timewindow):
            prices = close_price_list[s-timewindow:s]
            price_judge_high = high_price_list[s-timewindow:s+timewindow]
            price_judge_low = low_price_list[s-timewindow:s+timewindow]
            price_judge_open = open_price_list[s-timewindow:s+timewindow]
            price_judge_close = close_price_list[s-timewindow:s+timewindow]

            result = self.sample_judgement.judge(prices, 0.05)
            if result == None:
                continue
            result_list = []
            for m in range(0, len(self.feature_builder_list)):
                result_list.append(0)

            for mindex, m in enumerate(feature_builder_list):
                m.feature_build(price_judge_open, price_judge_high, price_judge_low, price_judge_close, mindex, result_list)
            if result_list[0] != 0:
                print result_list[0]
                print result
            samples.append(result_list)
            classes.append(result)
#        if len(samples) < 100:
#            sys.stderr.write("sample is too small\n")
#            sys.exit(1)
#
        regretor = LogisticRegression()
        regretor.fit(samples, classes)
        predict_value = regretor.predict_proba(samples)
        precision, recall, threshold = precision_recall_curve(numpy.array(classes), predict_value[:,0])
        print precision
        print recall
        print threshold
        area = auc(recall, precision)
        print "auc = %.4f" %(area)

if __name__ == "__main__":
    judger = prices_judgement()
    two_crow = twocrow_builder()
    feature_builder_list = []
    feature_builder_list.append(two_crow)
    model = base_model(feature_builder_list, judger)
    open_prices = []
    high_prices = []
    low_prices = []
    close_prices = []
    load_data(sys.argv[1], open_prices, high_prices, low_prices, close_prices)
    
    model.model_process(numpy.array(open_prices), numpy.array(high_prices), numpy.array(low_prices), numpy.array(close_prices), 14)
