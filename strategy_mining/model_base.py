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
    """
    from /home/work/workplace/pytrade/strategy_mining/utest_data/stocks/AAPL.csv to AAPL
    """
    return os.path.splitext(pathname.split("/")[-1])[0]

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

def get_all():
    sym2df = {}
    i = 0
    for each in get_file_list("/home/work/workplace/stock_data/"):
        symbol = get_stock_from_path(each)
        df = get_stock_data_pd(symbol)
        df = cal_features(df)
        df = judge(df)
        sym2df[symbol] = df #.dropna()
        i += 1
        #if i > 5:
        #    break
    return sym2df

def cal_features(df):
    builders = build_features()
    for mindex, m in enumerate(builders):
        feat = m.feature_build(df['open'].values,
                        df['high'].values,
                        df['low'].values,
                        df['close'].values,
                        df['adjclose'].values,
                        df['volume'].values,
                        mindex, 30)
        dates =  df.index.values
        assert feat.size == df.shape[0] and dates.size == feat.size
        #pdFeat = pd.DataFrame({"feat"+str(mindex):feat},index=dates)

        #assert pdFeat.shape[0] == df.shape[0]

        #df = df.merge(pdFeat, left_index=True, right_index=True, how='left')
        df['feat_'+str(mindex)] = feat
    return df
def get_stock_data_pd(symbol):
    names = ["date", 'open', 'high', 'low', 'close','volume', 'adjclose']
    df = pd.read_csv("/home/work/workplace/stock_data/"+symbol+".csv", names = names, skiprows = 1,  sep=",", index_col = 'date', parse_dates=True).sort_index()
    df['volume'] = df['volume']*1.0
    return df


def _judge(df, window):
    df["close_shift"] = df["close"].shift(-1 * window)
    df["label" + str(window)] = df["close_shift"]/df["close"]
    return df 

def judge(df):
    df = _judge(df, 1)
    df = _judge(df, 2)
    df = _judge(df, 3)
    df = _judge(df, 5)
    df = _judge(df, 6)
    df = _judge(df, 8)
    df = _judge(df, 10)
    df = _judge(df, 20)
    df = _judge(df, 30)
    df = _judge(df, 60)
    return df

def get_stock_data(filename, str_startdate = None, str_utildate = None, length = 1000):
    """
    input filename : the path of stock daily data
    """
    if str_utildate == None:
        str_utildate = get_date_str()
    dt_utildate = parse_date_str(str_utildate)
    dt_startdate = parse_date_str(str_startdate)
    dates = []
    open_prices = []
    high_prices = []
    low_prices = []
    close_prices = []
    adjust_close_prices = []
    volumes = []
    load_data(filename, dates, open_prices, high_prices, low_prices, close_prices, adjust_close_prices, volumes)
    dates.reverse()
    open_prices.reverse()
    high_prices.reverse()
    low_prices.reverse()
    close_prices.reverse()
    adjust_close_prices.reverse()
    volumes.reverse()
    dates2 = [] 
    open_prices2 = []
    high_prices2 = []
    low_prices2 = []
    close_prices2 = []
    adjust_close_prices2 = []
    volumes2 = []
    for i in range(-1 * len(dates), 0):
        dt_cur_date = parse_date_str(dates[i])
        if dt_cur_date < dt_utildate and dt_cur_date >= dt_startdate:
            dates2.append(dates[i])
            open_prices2.append(open_prices[i])
            high_prices2.append(high_prices[i])
            low_prices2.append(low_prices[i])
            close_prices2.append(close_prices[i])
            adjust_close_prices2.append(adjust_close_prices[i])
            volumes2.append(volumes[i])
    if len(dates2) < length:
        length = len(dates)
    return  dates2[len(dates2)-length:len(dates2)+1], \
            open_prices2[len(open_prices2)-length:len(open_prices2)+1], \
            high_prices2[len(high_prices2)-length:len(high_prices2)+1], \
            low_prices2[len(low_prices2)-length:len(low_prices2)+1], \
            close_prices2[len(close_prices2)-length:len(close_prices2)+1], \
            adjust_close_prices2[len(adjust_close_prices2)-length:len(adjust_close_prices2)+1], \
            volumes2[len(volumes2)-length:len(volumes2)+1]

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
    #print get_stock_data('/home/work/workplace/stock_data/MSFT.csv', '2010-01-01', '2010-02-01')
    #print get_all()
    df = get_stock_data_pd("MSFT")
    #cal_features(df)
    judge(df, 1)
