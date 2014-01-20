#!/usr/bin/env python
import sys,os,urllib2,time
from multiprocessing import Pool
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

from pyalgotrade.tools import yahoofinance;

root = "/home/work/stock_data/"

def get_sp500():#{{{
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
    return symbols  
#}}}

def one_work(symbol):
    print symbol
    retry = 3
    while retry >= 0:
        try:
            year = time.strftime('%Y',time.localtime(time.time()))
            year = int(year)
            yahoofinance.download_daily_bars(symbol, year, os.path.join(root, symbol + ".csv.bk"))
            os.rename(os.path.join(root,symbol+".csv.bk"), os.path.join(root,symbol+".csv"))
        except Exception,ex:
            print symbol, Exception, ":", ex
            retry -=1
            continue
        break

def main():
    symbols = get_sp500()
    pool = Pool(processes =100)
    result = {}
    for symbol in symbols:
        result[symbol] = pool.apply_async(one_work, (symbol,))
    for symbol in symbols:
        result[symbol].get()

if __name__ == '__main__':
    main()
