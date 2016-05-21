#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author 
import sys,os
import datetime
import pandas as pd
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

def get_file_list(rootdir):
    file_list = []
    for f in os.listdir(rootdir):
        if f == None or not f.endswith(".csv"):
            continue
        file_list.append(os.path.join(rootdir, f))
         
    return file_list

def get_stock_from_path(pathname):
    return os.path.splitext(os.path.split(pathname)[-1])[0]

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

