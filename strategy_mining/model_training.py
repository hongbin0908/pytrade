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
from volume_pattern import *
from sklearn import cross_validation
from  sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import neural_network
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor 
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
            sys.stderr.write("exception: %s\n" %(e))
            return [0]
        tmp_sample = numpy.nan_to_num(numpy.array(samples))
        if numpy.all(tmp_sample == 0):
            return [0]
        predict_value = self.model_predictor.predict(tmp_sample)
        if predict_value > 1.12:
            return [predict_value]
        else:
            return [0]

    def dump_model(self):
        # TODO
        pass        

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
   feature_builder_ohc(talib.ADXR, feature_builder_list)#   feature_builder_ohc(talib.CCI, feature_builder_list)
   feature_builder_ohc(talib.CCI, feature_builder_list)
   feature_builder_ohc(talib.MINUS_DI, feature_builder_list)
   feature_builder_ohc(talib.PLUS_DI, feature_builder_list)
   feature_builder_ohc(talib.WILLR, feature_builder_list)
   feature_builder_volume(talib.ADOSC, feature_builder_list)
   feature_builder_volume(talib.AD, feature_builder_list)
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
   
#   model_predictor = RandomForestRegressor(n_estimators=5) 
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
       if len(open_prices) < 7:
           continue
       model.build_sample(numpy.array(open_prices[:-7]), numpy.array(high_prices[:-7]), numpy.array(low_prices[:-7]), numpy.array(close_prices[:-7]), numpy.array(adjust_close[:-7]), numpy.array(volume[:-7]), 7)
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
       if result == None:
           continue
       if result[0] >1.12 and result[0] > 0:
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
