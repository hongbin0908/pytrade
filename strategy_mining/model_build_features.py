#!/usr/bin/env python
#@author redbin@outlook.com

import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

import numpy as np

# the classfier
from price_judgement import prices_judgement2
from model_traing_features import *
from model_base import get_file_list,get_stock_data,get_stock_from_path,format10,feature_builder_ohc

# parse the command paramaters
from optparse import OptionParser

import logging

import subprocess


logging.basicConfig(level = logging.DEBUG,
    format = '%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s',                                                                              
    filename = 'log/%s.log' % (os.path.basename(sys.argv[0]),), 
    filemode = 'a')
def main(options, args):
    judger = prices_judgement2();

    feature_builder_list = eval(options.features)()
    # writ the features name into "samplex.head"
    fsamplex_head = file("samplesX.head", "w")
    fsamplex_head_with_sym = file("samplesX_with_sym.head", "w")
    print >> fsamplex_head_with_sym, "%s,%s," % ("symbol", "date"), 
    for fun_feature in feature_builder_list:
        if fun_feature == feature_builder_list[-1]:
            print >> fsamplex_head, "%s" % fun_feature.name()
            print >> fsamplex_head_with_sym, "%s" % fun_feature.name()
        else:
            print >> fsamplex_head, "%s," % fun_feature.name(),
            print >> fsamplex_head_with_sym, "%s," % fun_feature.name(),
    fsamplex_head.close()
    fsamplex_head_with_sym.close()


    file_list = get_file_list("/home/work/workplace/stock_data/") # get the stock file list
    if True == options.short_data:
        short_num = 10
        file_list = file_list[0:short_num]
        logging.info('using --short_data param to load only %d stocks' % short_num)
    dic_sym2samples = {} # the feature list 
    dic_sym2samples_with_sym = {} # the feature list 
    dic_sym2classes = {} # the class results
    size = 0
    np_samples_all = None
    np_samples_all_with_sym = None
    stock_num = 0
    f_sample_prices = file("samples_prices.csv", "w")
    f_sampleY = file(options.train, "w")
    f_last = file(options.last, "w")
    for s in file_list:
        stock_num += 1
        if stock_num % 10  == 0:
            logging.debug("build the %d's stocks" % stock_num);
        s_stock_symbol = get_stock_from_path(s)
        l_samples = [] #  the cur stock's samples
        l_classes = [] #  the cur stock's classes
        dates, open_prices, high_prices, low_prices, close_prices, adjust_close_prices,volumes = get_stock_data(s)
        if len(open_prices) < 499:
            logging.debug("%s is too short(%d)!" % (s, len(open_prices) ) )
            continue
        # normalize the price
        format10(open_prices, high_prices,low_prices,close_prices,adjust_close_prices)
        for s in range(0, len(dates)):
            print >> f_sample_prices, "%s,%s,%f,%f,%f,%f,%f,%d" % \
                (s_stock_symbol, dates[s],open_prices[s],high_prices[s],low_prices[s],close_prices[s],adjust_close_prices[s],volumes[s])
        features = []
        for mindex, m in enumerate(feature_builder_list):
            features.append(m.feature_build(np.array(open_prices),
                                            np.array(high_prices),
                                            np.array(low_prices),
                                            np.array(close_prices),
                                            np.array(adjust_close_prices),
                                            np.array(volumes), mindex, 7).tolist())
        rawSamples = np.column_stack(features)

        l_dates_valid = []
        l_classes_valid = []
        l_classesY_valid = []
        assert(len(dates) == rawSamples.shape[0])
        for s in range(0, rawSamples.shape[0]):
            if (not np.isnan(np.min(rawSamples[s]))  \
                    and s < rawSamples.shape[0]-2):
                """ if no nan value or is not the last one """
                l_samples.append(rawSamples[s])
                l_dates_valid.append(dates[s])
                if close_prices[s+1] > close_prices[s]:
                    l_classesY_valid.append(1)
                else:
                    l_classesY_valid.append(0)
        np_classesY_valid = np.array(l_classesY_valid)
        np_classesY_valid.resize(np_classesY_valid.shape[0], 1)
        np_samples = np.array(l_samples)
        np_dates_valid = np.array(l_dates_valid)
        np_dates_valid.resize(np_dates_valid.shape[0],1)
        l_syms = []
        for i in range(np_dates_valid.shape[0]):
            l_syms.append(s_stock_symbol)
        np_syms = np.array(l_syms)
        np_syms.resize(np_dates_valid.shape[0],1)

        for i in range(np_samples.shape[0]):
            for j in range(np_samples.shape[1]):
                print np_samples[i,j]
                print >> f_sampleY, "%f," % np_samples[i,j],
            print >> f_sampleY, "%d"  % np_classesY_valid[i,0]
        print >> f_last, "%s,%s," % (dates[-1],s_stock_symbol) ,
        for j in range(np_samples.shape[1]):
            if j != np_samples.shape[1]-1:
                print >> f_last, "%f," % rawSamples[-1,j],
            else:
                print >> f_last, "%f" % rawSamples[-1,j]
    f_sample_prices.close()
    f_sampleY.close()
    f_last.close()
    cmd = subprocess.Popen("cat samplesX.head samplesX.csv > samplesX_with_head.csv", shell=True)
    cmd.wait()
        
    
def parse_options(paraser):
    """
    parser command line
    """
    parser.add_option("--short_data", dest="short_data",action = "store_true", default=False, help = "only pick up the 100 stock to test purpose")
    parser.add_option("--train", dest="train",action = "store", default="data/train.csv", help = "the output filename")
    parser.add_option("--last", dest="last",action = "store", default="data/last.csv", help = "the output filename")
    parser.add_option("--features", dest="features", action = "store", default="build_features", help="the features select class")
    return parser.parse_args()
if __name__ == "__main__":
    parser = OptionParser()
    (options, args) = parse_options(parser)
    main(options, args)
