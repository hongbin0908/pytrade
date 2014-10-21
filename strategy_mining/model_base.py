#!/usr/bin/env python
#@author 
import sys,os
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

def get_stock_data(filename):
    """
    input filename : the path of stock daily data
    """
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
    return dates[-1*length:], open_prices[-1*length:], high_prices[-1*length:], low_prices[-1*length:], close_prices[-1*length:], adjust_close_prices[-1*length:], volumes[-1*length:]
if __name__ == '__main__':
    main()
