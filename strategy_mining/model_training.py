#-*-encoding:gbk-*-

import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn import metrics
from price_judgement import *
from two_crow_builder import *
from three_inside_pattern import *
from three_inside_strike import *
from sklearn import linear_model
import numpy
from sklearn import tree
from other_pattern import *
from momentum_pattern import *
from sklearn import cross_validation
from  sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor 
class base_model:
    #定义基本属性
    name = "base_model"
    #特征生成方法列表，其中，每一个特征的参数形式均为(prices_list, index,feature_result_list),其中，index表示该特征生成的下标，生成的结果会存放在feature_result_list下
    feature_builder_list = []
    sample_judgement = None
    model_predictor = None
    samples = []
    classes = []
    int_num = 0
    def __init__(self, feature_builder_list_input, sample_judgement_input, model_predictor_input):
        self.feature_builder_list = feature_builder_list_input
        self.sample_judgement = sample_judgement_input
        self.model_predictor = model_predictor_input

    def build_sample(self, open_price_list, high_price_list, low_price_list, close_price_list, adjust_close_list, volume_list, timewindow):
        self.int_num = self.int_num + 1
        samples = []
        for mindex, m in enumerate(self.feature_builder_list):
            samples.append(m.feature_build(open_price_list, high_price_list, low_price_list, close_price_list, adjust_close_list, volume_list, mindex, timewindow).tolist())
        tmp_array = numpy.nan_to_num(numpy.column_stack(samples))
        tmp_prices = self.sample_judgement.judge(adjust_close_list, 0.05, 7)
        #判断是否是无效的
#        print "before=", self.samples.shape
        for s in range(0, tmp_array.shape[0]):
            if (numpy.any(tmp_array[s]!=0 )) and (tmp_prices[s] != -2 ):
                self.samples.append(tmp_array[s])
                self.classes.append(tmp_prices[s])
#                tmp_list = []
#                tmp_price = []
#                tmp_list.append(tmp_array[s])
#                tmp_price.append(tmp_prices[s])
#                if self.samples.shape[0] == 0:
#                    self.samples = numpy.array(tmp_list)
#                    self.classes = numpy.array(tmp_price)
#                else:
#                    tmp_samples = numpy.append(self.samples, tmp_list, axis=0 )
#                    self.samples = tmp_samples
#                    self.classes = numpy.append(self.classes, tmp_price, axis = 1)
        row_num = len(self.feature_builder_list)
        if self.int_num%10 == 0:
           print self.int_num
    def post_process(self):
        tmp_samples = numpy.array(self.samples)
        self.samples = tmp_samples
        tmp_prices = numpy.array(self.classes)
        self.classes = tmp_prices
        print self.samples.shape
#    def build_sample(self, open_price_list, high_price_list, low_price_list, close_price_list, adjust_close_list, volume_list, timewindow):
#        for s in range(timewindow, len(open_price_list)-timewindow):
#            prices = close_price_list[s-timewindow:s+timewindow]
#            price_judge_high = high_price_list[s-timewindow:s]
#            price_judge_low = low_price_list[s-timewindow:s]
#            price_judge_open = open_price_list[s-timewindow:s]
#            price_judge_close = close_price_list[s-timewindow:s]
#            adjust_close = adjust_close_list[s-timewindow:s]
#            volume = volume_list[s-timewindow:s]
#            result = self.sample_judgement.judge(prices, 0.05)
#            if result == None:
#                continue
#            result_list = []
#            for m in range(0, len(self.feature_builder_list)):
#                result_list.append(-1)
#
#            for mindex, m in enumerate(self.feature_builder_list):
#                m.feature_build(price_judge_open, price_judge_high, price_judge_low, price_judge_close, adjust_close, volume, mindex, result_list)
#                #print result_list
#            if result_list.count(0) == len(result_list):
#                continue
#            self.samples.append(result_list)
#            self.classes.append(result)
#            tmp_str = str(result)
#            for s in result_list:
#                tmp_str = tmp_str + "\t" + str(s)
#            if result_list[0] != 0:
#                print tmp_str

    def model_process(self):
        if len(self.samples) == 0:
            return
        self.model_predictor.fit(self.samples, self.classes)
        predict_value = self.model_predictor.predict(self.samples)
        r2_score = metrics.r2_score(self.classes, predict_value)
        print r2_score
#        predict_value = self.model_predictor.score_samples(self.samples)
#        predict_value = self.model_predictor.predict(self.samples)
#        print numpy.array(tmp_test_class).shape
#        print self.model_predictor.class_prior_
#        precision, recall, threshold = roc_curve(self.classes, predict_value)
        
#        area = auc(recall, precision)
#        print "auc = %.4f" %(area)
#        for s, s_test in bs:
#            tmp_list = []
#            tmp_test = []
#            tmp_class = []
#            tmp_test_class = []
#            for m in s:
#                tmp_list.append(self.samples[m])
#                tmp_class.append(self.classes[m])
#            for m in s_test:
#                tmp_test.append(self.samples[m])
#                tmp_test_class.append(self.classes[m])
#            self.model_predictor.fit(numpy.array(tmp_list), numpy.array(tmp_class))
#            predict_value = self.model_predictor.predict_proba(numpy.array(tmp_test))
#            print numpy.array(tmp_test_class).shape
#            print predict_value.shape
#            precision, recall, threshold = roc_curve(numpy.array(tmp_test_class), predict_value[:,0])
#            
#            area = auc(recall, precision)
#            print "auc = %.4f" %(area)

    def result_predict(self, open_price_list, high_price_list, low_price_list, close_price_list, adjust_close_list, volume_list, timewindow):
        samples = []
#        
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
            sys.stderr.write("exception: %s\n" %(e))
            return [0]
        tmp_sample = numpy.nan_to_num(numpy.array(samples))
        if numpy.all(tmp_sample == 0):
            return [0]
        predict_value = self.model_predictor.predict(tmp_sample)
        if predict_value > 1.04:
            return [predict_value]
        else:
            return [0]

    def dump_model(self):
        pass        

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

def get_predict_value():
   judger = prices_judgement()
   two_crow = twocrow_builder()
   three_inside = three_inside_up_builder()
   three_inside_strike = three_inside_strike_builder()
   three_outside_move = three_outside_move_builder()
   three_star_south = three_star_south_builder()
   three_ad_white_soldier = three_ad_white_soldier_builder()
   abandoned_baby = abandoned_baby_builder()
   three_ad_block = three_ad_block_builder()
   belt_hold = belt_hold_builder()
   break_away = breakaway_builder()
   conceal_baby = conceal_baby_swallow_builder()

   feature_builder_list = []
   feature_builder(talib.CDLCOUNTERATTACK, feature_builder_list)

   feature_builder(talib.CDLDARKCLOUDCOVER , feature_builder_list)
   feature_builder(talib.CDLDOJI , feature_builder_list)
   feature_builder(talib.CDLDOJISTAR , feature_builder_list)
   feature_builder(talib.CDLDRAGONFLYDOJI , feature_builder_list)
   feature_builder(talib.CDLENGULFING , feature_builder_list)
   feature_builder(talib.CDLEVENINGDOJISTAR , feature_builder_list)
   feature_builder(talib.CDLEVENINGSTAR , feature_builder_list)
   feature_builder(talib.CDLGAPSIDESIDEWHITE , feature_builder_list)

   feature_builder(talib.CDLGRAVESTONEDOJI  , feature_builder_list)
   feature_builder(talib.CDLHAMMER  , feature_builder_list)
   feature_builder(talib.CDLHANGINGMAN , feature_builder_list)
   feature_builder(talib.CDLHARAMI , feature_builder_list)
   feature_builder(talib.CDLHARAMICROSS , feature_builder_list)
   feature_builder(talib.CDLHIGHWAVE , feature_builder_list)
   feature_builder(talib.CDLHIKKAKE , feature_builder_list)
   feature_builder(talib.CDLHIKKAKEMOD , feature_builder_list)


   feature_builder(talib.CDLIDENTICAL3CROWS , feature_builder_list)
   feature_builder(talib.CDLINNECK , feature_builder_list)
   feature_builder(talib.CDLINVERTEDHAMMER , feature_builder_list)
   feature_builder(talib.CDLKICKING , feature_builder_list)
   feature_builder(talib.CDLKICKINGBYLENGTH , feature_builder_list)
   feature_builder(talib.CDLLADDERBOTTOM , feature_builder_list)
   feature_builder(talib.CDLLONGLEGGEDDOJI , feature_builder_list)
   feature_builder(talib.CDLLONGLINE , feature_builder_list)
   feature_builder(talib.CDLMARUBOZU , feature_builder_list)
   feature_builder(talib.CDLMATCHINGLOW , feature_builder_list)
   feature_builder(talib.CDLMATHOLD , feature_builder_list)
   feature_builder(talib.CDLMORNINGDOJISTAR , feature_builder_list)
   feature_builder(talib.CDLMORNINGSTAR , feature_builder_list)
   feature_builder(talib.CDLONNECK , feature_builder_list)
   feature_builder(talib.CDLPIERCING , feature_builder_list)

   feature_builder(talib.CDLRICKSHAWMAN , feature_builder_list)
   feature_builder(talib.CDLRISEFALL3METHODS , feature_builder_list)
   feature_builder(talib.CDLSEPARATINGLINES , feature_builder_list)
   feature_builder(talib.CDLSHOOTINGSTAR , feature_builder_list)
   feature_builder(talib.CDLSHORTLINE , feature_builder_list)
   feature_builder(talib.CDLSPINNINGTOP , feature_builder_list)
   feature_builder(talib.CDLSTALLEDPATTERN , feature_builder_list)

   feature_builder(talib.CDLSTICKSANDWICH , feature_builder_list)
   feature_builder(talib.CDLTAKURI , feature_builder_list)
   feature_builder(talib.CDLTASUKIGAP , feature_builder_list)
   feature_builder(talib.CDLTHRUSTING , feature_builder_list)
   feature_builder(talib.CDLTRISTAR , feature_builder_list)
   feature_builder(talib.CDLUNIQUE3RIVER , feature_builder_list)
   feature_builder(talib.CDLUPSIDEGAP2CROWS , feature_builder_list)

   feature_builder(talib.CDLXSIDEGAP3METHODS , feature_builder_list)
   feature_builder_ohc(talib.ADXR, feature_builder_list)
   feature_builder_list.append(two_crow)
   feature_builder_list.append(three_inside)
   feature_builder_list.append(three_inside_strike)
   feature_builder_list.append(three_outside_move)
   feature_builder_list.append(three_star_south)
   feature_builder_list.append(three_ad_white_soldier)
   feature_builder_list.append(abandoned_baby)
   feature_builder_list.append(three_ad_block)
   feature_builder_list.append(belt_hold)
   feature_builder_list.append(break_away)
   feature_builder_list.append(conceal_baby)
   
   model_predictor = GradientBoostingRegressor() 
   model = base_model(feature_builder_list, judger, model_predictor)
   all_open_prices = []
   all_high_prices = []
   all_low_prices = []
   all_close_prices = []
   all_adjust_close = []
   all_volume = []
   file_list = get_file_list()
   stock_name = []
   for s in file_list:
       open_prices = []
       high_prices = []
       low_prices = []
       close_prices = []
       adjust_close = []
       volume = []
       load_data(s, open_prices, high_prices, low_prices, close_prices, adjust_close, volume)
       open_prices.reverse()
       high_prices.reverse()
       low_prices.reverse()
       close_prices.reverse()
       adjust_close.reverse()
       volume.reverse()
       model.build_sample(numpy.array(open_prices), numpy.array(high_prices), numpy.array(low_prices), numpy.array(close_prices), numpy.array(adjust_close), numpy.array(volume), 14)
   model.post_process()
   model.model_process()
   num  = 0
   for s in file_list:
       open_prices = []
       high_prices = []
       low_prices = []
       close_prices = []
       adjust_close = []
       volume = []
       load_data(s, open_prices, high_prices, low_prices, close_prices, adjust_close, volume)
       open_prices.reverse()
       high_prices.reverse()
       low_prices.reverse()
       close_prices.reverse()
       adjust_close.reverse()
       volume.reverse()
       result = model.result_predict(numpy.array(open_prices), numpy.array(high_prices), numpy.array(low_prices), numpy.array(close_prices),
           numpy.array(adjust_close), numpy.array(volume), 7)
       if result[0] > 1.02:
           num = num + 1
           print "%s\t%.4f" %(s, result[0])
   print num

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
    get_predict_value()
