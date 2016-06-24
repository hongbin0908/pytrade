#!/usr/bin/env python2.7

import os, sys
import finsymbols
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

from main.yeod import engine
from main.utils import time_me

def get_dow():
    symbols = [ "AAPL", "AXP", "BA", "CAT", "CSCO",
                "CVX", "DD", "DIS", "GE", "GS", "HD",
                "IBM", "INTC", "JNJ", "JPM", "MCD",
                "MMM", "MRK", "MSFT", "NKE", "PFE",
                "TRV", "UNH", "UTX", "V", "VZ", "WMT",
                "XOM", ]
    return symbols

def get_sp500():
    symbols = []
    for each in finsymbols.symbols.get_sp500_symbols():
        symbols.append(each['symbol'].strip())
    return symbols

def get_data_root(target):
    data_root = os.path.join(root, 'data', 'yeod', target)
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    return data_root

@time_me
def main(argv):
    target = argv[0]
    pool_num = int(argv[1])
    symbols = eval("get_%s" % target)()
    return engine.work(symbols, get_data_root(target), pool_num)
    pass

if __name__ == '__main__':
    main(sys.argv[1:])
