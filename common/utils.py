#!/usr/bin/env python
# author hongbin0908@126.com
# some utils path or function 


import sys,os
import urllib2
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

def get_stock_data_path():
    """
    the location of the stock data of cvs format
    """
    return "/home/work/stock_data/"

def get_sp500():
    finviz_retry = 3
    while finviz_retry >= 0:
        try:
            resp = urllib2.urlopen("""http://finviz.com/export.ashx?v=152&f=idx_sp500&ft=1&ta=1&p=d&r=1&c=1""")
        except Exception,ex:
            print Exception,":",ex
            finviz_retry -= 1
            continue
        break
    symbols = [symbol.strip().strip("\"") for symbol in resp.read().split("\n")[1:]]
    return symbols[0:-1] 
