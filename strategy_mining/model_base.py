#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author 
import sys,os
import datetime
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

def main():
    pass


import talib
import numpy
import two_crow_builder

class feature_builder_ohc():
    def __init__(self, feature_func, builder_list):
        self.feature_build_func = feature_func
        builder_list.append(self)

    def feature_build(self, open_price, high_price, low_price, close_price, adjust_close, volume, index, feature_result_list):
        result = self.feature_build_func(high_price, low_price, close_price)
        return result

    def name(self):
        return self.feature_build_func.__name__
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

def get_stock_from_path(pathname):
    """
    from /home/work/workplace/pytrade/strategy_mining/utest_data/stocks/AAPL.csv to AAPL
    """
    return pathname.split("/")[-1].split(".")[0]

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

def get_stock_data(filename, str_utildate = None):
    """
    input filename : the path of stock daily data
    """
    if str_utildate == None:
        str_utildate = get_date_str()
    dt_utildate = parse_date_str(str_utildate)
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
    length = 365 * 2
    if len(dates) < length:
        length = len(dates)
    dates2 = [] 
    open_prices2 = []
    high_prices2 = []
    low_prices2 = []
    close_prices2 = []
    adjust_close_prices2 = []
    volumes2 = []
    for i in range(-1 * length, 0):
        dt_cur_date = parse_date_str(dates[i])
        if dt_cur_date < dt_utildate:
            dates2.append(dates[i])
            open_prices2.append(open_prices[i])
            high_prices2.append(high_prices[i])
            low_prices2.append(low_prices[i])
            close_prices2.append(close_prices[i])
            adjust_close_prices2.append(adjust_close_prices[i])
            volumes2.append(volumes[i])
        else:
            break
    return dates2, open_prices2, high_prices2, low_prices2, close_prices2, adjust_close_prices2, volumes2
def get_date_str(): # {{{
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d')
# }}}
def parse_date_str(date_str): # {{{
    return datetime.datetime.strptime(date_str, '%Y-%m-%d')
# }}}


def get_stock_data_one_day(sym, datestr,stock_root="/home/work/workplace/stock_data/"):
    """
    描述: 获取指定一天的股票数据
    输入: 股票代码 时间(YYYY-MM-DD)
    输出: dict {"open":xx,"high":xx,"low":xx,"close":xx,"adj_close":xx,"volume":xx}

    """
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

        # random check
        assert( res["low"] <= res["high"]  )
        assert( res["low"] <= res["open"]  )
        assert( res["low"] <= res["close"]  )
        
        break
    return res;

if __name__ == '__main__':
    main()
