#!/usr/bin/env python2.7

import os, sys
from datetime import date
from pyalgotrade.tools import yahoofinance
import multiprocessing
import finsymbols

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, "..", "data", 'yeod_full')
if not os.path.exists(root):
    os.makedirs(root)

def get_full():
    syms = set()
    for each in finsymbols.symbols.get_nasdaq_symbols():
        syms.add(each['symbol'])
    for each in finsymbols.symbols.get_nyse_symbols():
        syms.add(each['symbol'])
    return list(syms)

def one_work(symbol): 
    retry = 3
    while retry > 0:
        try:
            eod = yahoofinance.download_csv(symbol, date(1970,01,01),date(2099,01,01), 'd')
            with open(os.path.join(root,symbol+".csv"), 'w') as fout:
                print >> fout, eod
        except Exception,ex:
            print symbol, Exception, ":", ex
            retry -=1
            continue
        break

def main():
    symbols = get_full()
    print len(symbols)
    pool = multiprocessing.Pool(processes = 4 )
    result = {}
    for symbol in symbols:
        result[symbol] = pool.apply_async(one_work, (symbol,))
    for symbol in symbols:
        result[symbol].get()
if __name__ == '__main__':
    main()
