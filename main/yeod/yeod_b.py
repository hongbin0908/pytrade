#!/usr/bin/env python2.7

import os, sys
import finsymbols
import numpy as np
import pandas as pd
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

from main.yeod import engine
from main.utils import time_me

def get_MSFT():
    return ['MSFT']

def get_index_dow():
    return ['^DJI']

def get_dow():
    symbols = [ "AAPL", "AXP", "BA", "CAT", "CSCO",
                "CVX", "DD", "DIS", "GE", "GS", "HD",
                "IBM", "INTC", "JNJ", "JPM", "MCD",
                "MMM", "MRK", "MSFT", "NKE", "PFE",
                "TRV", "UNH", "UTX", "V", "VZ", "WMT",
                "XOM", ]
    return symbols

def get_sp500Top10():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i,each in df.head(10).iterrows()]

def get_sp500Top200():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i,each in df.head(200).iterrows()]
def get_sp500Top100():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i,each in df.head(100).iterrows()]

def get_data_root_batch(target,batch,num):
    data_root = os.path.join(root, 'data', 'yeod_batch', target+"-"+str(batch), str(num))
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    return data_root

@time_me
def main(argv):
    target = argv[0]
    batch = int(argv[1])
    pool_num = int(argv[2])
    symbols = eval("get_%s" % target)()
    idx = 0; num = 0
    while True:
        cur = symbols[idx:idx+batch]
        engine.work(cur, get_data_root_batch(target, batch, num), pool_num)
        print cur
        idx = idx+batch
        num += 1
        if idx + batch > len(symbols):
            break

if __name__ == '__main__':
    main(sys.argv[1:])
