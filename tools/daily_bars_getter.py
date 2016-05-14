#!/usr/bin/env python
import os, sys

local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")
sys.path.append(local_path + "/../common/")

import YahooFinceExted as yahoofinance;

import urllib, urllib2

import multiprocessing

import finsymbols


root = os.path.join(local_path, "..", "data", 'yeod')
if not os.path.exists(root):
    os.makedirs(root)

def get_nasdaq2000():
    fd = open(local_path+"/nasdaq_symbols", 'r')
    ss=[]
    for j in fd:
        stock_name = j.rstrip()
        ss.append(stock_name)
    return ss

def get_sp500():#{{{
    symbols = []
    for each in finsymbols.symbols.get_sp500_symbols():
        symbols.append(each['symbol'])
    return symbols  
#}}}

def one_work(symbol): # {{{
    print symbol
    retry = 3
    while retry >= 0:
        try:
            yahoofinance.download_daily_bars_full(symbol, os.path.join(root, symbol + ".csv.bk"))
            os.rename(os.path.join(root,symbol+".csv.bk"), os.path.join(root,symbol+".csv"))
        except Exception,ex:
            print symbol, Exception, ":", ex
            retry -=1
            continue
        break
#}}}

def main():
    symbols = get_sp500()
    #symbols = set()
    #for s in get_nasdaq2000():
    #    symbols.add(s)
    # symbols = list(symbols)
    pool = multiprocessing.Pool(processes =10 )
    result = {}
    for symbol in symbols:
        #one_work(symbol)
        result[symbol] = pool.apply_async(one_work, (symbol,))
    for symbol in symbols:
        result[symbol].get()

if __name__ == '__main__':
    main()
