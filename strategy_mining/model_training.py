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
from sklearn import preprocessing
from model_traing_features import *

class base_model:
    feature_builder_list = []
    sample_judgement = None
    model_predictor = None
    samples = []
    classes = []
    int_num = 0 # give the visual presition of the precess
    def __init__(self, feature_builder_list_input, sample_judgement_input, model_predictor_input, rootdir = "/home/work/workplace/stock_data/"):
        self.feature_builder_list = feature_builder_list_input
        self.sample_judgement = sample_judgement_input
        self.model_predictor = model_predictor_input
        self.normalizer = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
        self.rootdir = rootdir
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
            cur_feat = m.feature_build(open_price_list, high_price_list, low_price_list, close_price_list, adjust_close_list, volume_list, mindex, timewindow).tolist()
            samples.append(cur_feat)
        tmp_array = numpy.nan_to_num(numpy.column_stack(samples))
        tmp_array = numpy.column_stack(samples)

        print tmp_array.shape; 

        tmp_prices = self.sample_judgement.judge(open_price_list, high_price_list, low_price_list, close_price_list, 0.05, 7)
        total = 0; valid = 0

        for s in range(0, tmp_array.shape[0]):
            total += 1
            if (not numpy.isnan(numpy.min(tmp_array[s]))) and (not numpy.isnan(tmp_prices[s])):
                valid +=1
                self.samples.append(tmp_array[s])
                self.classes.append(tmp_prices[s])
        #print "DEBUG: loaded %d lines, valid:%d lines" % (total, valid)
    def post_process(self):
        tmp_samples = numpy.array(self.samples)
        self.samples = tmp_samples
        tmp_prices = numpy.array(self.classes)
        self.samples = self.normalizer.fit_transform(tmp_samples, tmp_prices)
        self.classes = tmp_prices
        print "TRACE sample space: ", self.samples.shape

    def model_process(self):
        if len(self.samples) == 0:
            return
        train_size = self.train_len
        assert self.train_len + self.test_len == len(self.samples)
        print "DEBUG: train size: %d, predict size: %d" % (train_size, len(self.samples) - train_size)
        self.model_predictor.fit(self.samples[0:train_size], self.classes[0:train_size])
        print "TRACE training ... "
        predict_value = self.model_predictor.predict(self.samples[train_size:])
        print "predict_value", predict_value[0]
        print "TRACE train complete!"
        pos = 0; neg = 0; pos_all = 0; neg_all = 0; all = 0
        tests = self.classes[train_size:]
        for i in range(0, predict_value.size):
            if (predict_value[i]>1):
                if tests[i] > 1:
                    pos +=1
                pos_all+=1
            if (predict_value[i]<1):
                if tests[i] < 1:
                    neg +=1
                neg_all+=1
            all += 1
        opos  = 0
        for i in range(0, predict_value.size):
            if tests[i] > 1:
                opos += 1

        print "%d\t%d\t%f\t%d\t%d\t%f\t%f" % ( pos, pos_all, pos*1.0/pos_all, neg, neg_all, neg*1.0/neg_all, opos*1.0/all)
        
        r2_score = metrics.r2_score(self.classes[train_size:], predict_value)
        print "TRACE r2_score:", r2_score
    def model_test(self, timewindow):
        
        file_list = get_file_list(self.rootdir)

        for backdays in range(0, 30):
            final_list = []
            predicts = {} # stockname -> predict price change rate
            actuals  = {} # stockname -> actual  price change rate
            for s in file_list:
                open_prices, high_prices, low_prices, close_prices, adjust_close_prices, volume = get_stock_data(s)
                if len(open_prices) < 60:
                    continue
                pend = -7 - backdays
                open_prices_r = open_prices[0:pend]
                high_prices_r = high_prices[0:pend]
                low_prices_r  = low_prices[0:pend]
                close_prices_r = close_prices[0:pend]
                adjust_close_prices_r = adjust_close_prices[0:pend]
                volum_close_prices_r = volume[0:pend]
                actuals[get_stock_from_path(s)] = close_prices[backdays -1] / close_prices[pend -1]
                result = self.result_predict(numpy.array(open_prices_r),
                                             numpy.array(high_prices_r),
                                             numpy.array(low_prices_r),
                                             numpy.array(close_prices_r),
                                             numpy.array(adjust_close_prices_r),
                                             numpy.array(volum_close_prices_r),
                                             7)
                if result == None:
                    print "ERROR: model.result_predict of %s error" % s
                    continue
                if result > 0:
                    final_list.append((s, result))
                final_sort_list = sorted(final_list, key=lambda x:x[1], reverse = True)
        
                for s in final_sort_list[0:10]:
                    predicts[get_stock_from_path(s[0])] = s[1]
                for s in predicts:
                    print "%s %f %f" % (s, predicts[s], actuals[s])
        
        
    def result_predict(self, open_price_list, high_price_list, low_price_list, close_price_list, adjust_close_list, volume_list, timewindow):
        samples = []
        if len(open_price_list) < timewindow:
            print "ERROR: the size of input is less than %d" % (timewindow)
            return -1      
        if timewindow < 30:
            timewindow = 30
        
        open_price = open_price_list[-timewindow-2: ]
        print open_price; assert False
        high_price = high_price_list[-timewindow-2: ]
        low_price = low_price_list[-timewindow-2: ]
        close_price = close_price_list[-timewindow - 2 : ]
        adjust_price = adjust_close_list[-timewindow - 2 : ]
        volume = volume_list[-timewindow - 2 : ]
        try:
            for index, s in enumerate(self.feature_builder_list):
                samples.append(s.feature_build(open_price_list, high_price_list, low_price_list, close_price_list, adjust_close_list, volume_list, index, timewindow)[-1])
        except Exception,e:
            sys.stderr.write("ERROR exception: %s\n" %(e))
            return [0]
        tmp_sample = self.normalizer.transform(numpy.nan_to_num(numpy.array(samples)))
        if numpy.all(tmp_sample == 0):
            print "ERROR: the sample data are all zero!"
            return [0]
        predict_value = self.model_predictor.predict(tmp_sample)
        return predict_value[0]

    def dump_model(self):
        # TODO
        pass        

def main():
    judger = prices_judgement()
    feature_builder_list = build_features()
    model_predictor = GradientBoostingRegressor(n_estimators=40)
    model = base_model(feature_builder_list, judger, model_predictor)
    file_list = get_file_list(model.rootdir)
    print "TRACE loading stock .. "
    idx = 0
    for s in file_list:
        dates, open_prices, high_prices, low_prices, close_prices, adjust_close_prices,volume = get_stock_data(s, '1970-01-01', '2013-01-01')
        if len(open_prices) < 30:
            continue
        model.build_sample(numpy.array(open_prices[:-7]),
                           numpy.array(high_prices[:-7]),
                           numpy.array(low_prices[:-7]),
                           numpy.array(close_prices[:-7]),
                           numpy.array(adjust_close_prices[:-7]),
                           numpy.array(volume[:-7]), 7)
        idx += 1
        if idx > 10: 
            break
    model.train_len = len(model.samples)
    idx = 0
    for s in file_list:
        dates, open_prices, high_prices, low_prices, close_prices, adjust_close_prices,volume = get_stock_data(s, '2013-01-01', '2099-01-01')
        if len(open_prices) < 30:
            continue
        model.build_sample(numpy.array(open_prices[:-7]),
                           numpy.array(high_prices[:-7]),
                           numpy.array(low_prices[:-7]),
                           numpy.array(close_prices[:-7]),
                           numpy.array(adjust_close_prices[:-7]),
                           numpy.array(volume[:-7]), 7)
        idx += 1
        if idx > 10: 
            break
    model.test_len = len(model.samples) - model.train_len


    print "TRACE loaded stock: %d" % model.int_num
    model.post_process()
    model.model_process()
    #model.model_test(7)
    num  = 0
    final_list = []
    for s in file_list:
        dates, open_prices, high_prices, low_prices, close_prices, adjust_close_prices,volume = get_stock_data(s, '1970-01-01','2099-01-01')
        if len(open_prices) < 30:
            continue
        result = model.result_predict(numpy.array(open_prices), numpy.array(high_prices), 
                numpy.array(low_prices), numpy.array(close_prices),numpy.array(adjust_close_prices), 
                numpy.array(volume), 7)
        if result == None:
            print "ERROR: model.result_predict of %s error" % s
            continue
        if result > 0:
            final_list.append((s, result))
    final_sort_list = sorted(final_list, key=lambda x:x[1], reverse = True)
    for s in final_sort_list[0:10]:
       print "up", get_stock_from_path(s[0]), s[1]
    for s in final_sort_list[-10:]:
       print "down", get_stock_from_path(s[0]), s[1]
       
def get_stock_from_path(pathname):
    """
    from /home/work/workplace/pytrade/strategy_mining/utest_data/stocks/AAPL.csv to AAPL
    """
    return pathname.split("/")[-1].split(".")[0]
def get_file_list(rootdir):
    """hongbin0908@126.com
    a help function to load test data.
    """
    file_list = []
    for f in os.listdir(rootdir):
        if f == None or not f.endswith(".csv"):
            continue
        file_list.append(os.path.join(rootdir, f))
         
    return file_list
if __name__ == "__main__":
    main()
