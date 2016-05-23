#!/usr/bin/env python
import os, sys
import YahooFinceExted as yahoofinance
import multiprocessing
import finsymbols

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, "..", "data", 'yeod')
if not os.path.exists(root):
    os.makedirs(root)

def get_nasdaq():
    syms =  finsymbols.get_nasdaq_symbols()
    print len(syms)
    print syms[0]
    return syms

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
    pool = multiprocessing.Pool(processes =10 )
    result = {}
    for symbol in symbols:
        result[symbol] = pool.apply_async(one_work, (symbol,))
    for symbol in symbols:
        result[symbol].get()
if __name__ == '__main__':
    #get_nasdaq()
    main()
