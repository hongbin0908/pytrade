#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author 
import sys,os
import datetime
import numpy as np
import pandas as pd
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")
from model_traing_features import *

def main():
    pass


import talib
import numpy
import two_crow_builder

def get_file_list(rootdir):
    file_list = []
    for f in os.listdir(rootdir):
        if f == None or not f.endswith(".csv"):
            continue
        file_list.append(os.path.join(rootdir, f))
         
    return file_list

def get_stock_from_path(pathname):
    return os.path.splitext(os.path.split(pathname)[-1])[0]

def load_data(filename, dates, open_price, high_price, low_price, close_price, adjust_price, volume_list):
    fd = open(filename, "r")
    for j in fd:
        try:
            line_list = j.rstrip().split(",")
            date_str = line_list[0]
            open_p = float(line_list[1])
            volume = float(line_list[5])
            high_p = float(line_list[2])
            low_p = float(line_list[3])
            close_p = float(line_list[4])
            dates.append(date_str)
            open_price.append(open_p)
            high_price.append(high_p)
            low_price.append(low_p)
            close_price.append(close_p)
            volume = float(line_list[5])
            adjust_p = float(line_list[6])
            adjust_price.append(adjust_p)
            volume_list.append(volume)
        except Exception, e:
            continue
    fd.close()
    return 0


def format10(open_prices, high_prices, low_prices, close_prices, adjust_prices):
    """
    normalize the stock price. 
    firstly, set the open price of the first day to 1
    secondly, every price of evuery day normalized orcording the price
    """
    open_price_first = open_prices[0]
    for s in range(len(open_prices)):
        open_prices[s] = open_prices[s]/open_price_first
        high_prices[s] = high_prices[s]/open_price_first
        low_prices[s]  = low_prices[s] /open_price_first
        close_prices[s]= close_prices[s]/open_price_first
        adjust_prices[s]=adjust_prices[s]/open_price_first

def _judge(df, window):
    df["close_shift"] = df["close"].shift(-1 * window)
    df["label" + str(window)] = df["close_shift"]/df["close"]
    return df 

def judge(df):
    df = _judge(df, 1)
    df = _judge(df, 2)
    df = _judge(df, 3)
    df = _judge(df, 4)
    df = _judge(df, 5)
    df = _judge(df, 6)
    df = _judge(df, 8)
    df = _judge(df, 10)
    df = _judge(df, 20)
    df = _judge(df, 30)
    df = _judge(df, 60)
    return df
def get_all():
    sym2df = {}
    i = 0
    for each in get_file_list(os.path.join(local_path, '..', 'data','ta')):
        symbol = get_stock_from_path(each)
        df = get_stock_data_pd(symbol)
        #sym2df[symbol] = judge(df) #.dropna()
        sym2df[symbol] = df #.dropna()
        sym2
        i += 1
        #if i > 5:
        #    break
    return sym2df

def get_stock_data_pd(symbol):
    df = pd.read_csv(os.path.join(local_path, '..', 'data', 'ta', symbol+".csv"),  index_col = 'date', parse_dates=True).sort_index()
    return df



def get_date_str(): # {{{
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d')
# }}}
def parse_date_str(date_str): # {{{
    return datetime.datetime.strptime(date_str, '%Y-%m-%d')
# }}}


def get_stock_data_one_day(sym, datestr,stock_root="/home/work/workplace/stock_data/"):
    if len(datestr) != 10:
        assert(False)
    res = {}
    filename = os.path.join(stock_root, sym + ".csv")
    if not os.path.exists(filename): 
        print "filename not exists : " , filename
        assert(False)
    lines = open(filename, "r").readlines() ; assert(len(lines)>1)
    for line in lines[1:]:
        terms = line.rstrip().split(",")
        strdate = terms[0]
        if  len(strdate) != 10:
            print "the date string format error[date:%s][line:%s]" % (strdate, line)
            assert(False)
        if strdate != datestr :
            continue
        res["open"]  = float(terms[1])
        res["high"]  = float(terms[2])
        res["low"]   = float(terms[3])
        res["close"] = float(terms[4])
        res["volume"]  = float(terms[5])
        res["adj_close"] = float(terms[6])

        assert( res["low"] <= res["high"]  )
        assert( res["low"] <= res["open"]  )
        assert( res["low"] <= res["close"]  )
        
        break
    return res

def get_stock_data_span_day(sym, datestr,span, stock_root="/home/work/workplace/stock_data/"):
    if len(datestr) != 10:
        assert(False)
    res = {}
    filename = os.path.join(stock_root, sym + ".csv")
    if not os.path.exists(filename): 
        print "filename not exists : " , filename
        assert(False)
    lines = open(filename, "r").readlines() ; assert(len(lines)>1)
    index = 0
    for index in range(1, len(lines)):
        line = lines[index]
        terms = line.rstrip().split(",")
        strdate = terms[0]
        if  len(strdate) != 10:
            print "the date string format error[date:%s][line:%s]" % (strdate, line)
            assert(False)
        if strdate != datestr :
            continue
        line = lines[index + span]
        terms = line.rstrip().split(",")
        res["open"]  = float(terms[1])
        res["high"]  = float(terms[2])
        res["low"]   = float(terms[3])
        res["close"] = float(terms[4])
        res["volume"]  = float(terms[5])
        res["adj_close"] = float(terms[6])
        res["date"] = terms[0]

        # random check
        assert( res["low"] <= res["high"]  )
        assert( res["low"] <= res["open"]  )
        assert( res["low"] <= res["close"]  )
        
        break
    return res;
if __name__ == '__main__':
    print get_file_list(os.path.join(local_path, '..', 'data', 'yeod'))
    print get_stock_from_path("'C:\\pythonwp\\pytrade\\strategy_mining\\..\\data\\yeod\\A.csv")
    #print get_stock_data('/home/work/workplace/stock_data/MSFT.csv', '2010-01-01', '2010-02-01')
    #print get_all()
    #df = get_stock_data_pd("MSFT")
    #cal_features(df)
    #judge(df, 1)
