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

def get_sp500Top100():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i,each in df.head(100).iterrows()]
def get_sp500Top50():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i,each in df.head(50).iterrows()]

def get_sp500Top30():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i,each in df.head(30).iterrows()]
def get_sp500Top10():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i,each in df.head(10).iterrows()]
def get_sp500Top1020():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i,each in df.head(20).tail(10).iterrows()]
def get_sp500Top2030():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i,each in df.head(30).tail(10).iterrows()]
def get_sp500Top3040():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i,each in df.head(40).tail(10).iterrows()]
def get_sp500Top4050():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i,each in df.head(50).tail(10).iterrows()]
def get_sp500():
    symbols = []
    for each in finsymbols.symbols.get_sp500_symbols():
        symbols.append(each['symbol'].strip())
    return symbols

def get_sp500_energy():
    return [each['symbol'].strip() for each in finsymbols.symbols.get_sp500_symbols() if each['sector'] == 'Energy']

def get_data_root(target):
    data_root = os.path.join(root, 'data', 'yeod', target)
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    return data_root

@time_me
def main(argv):
    print "xxxx"
    target = argv[0]
    pool_num = int(argv[1])
    symbols = eval("get_%s" % target)()
    return engine.work(symbols, get_data_root(target), pool_num)
    pass

if __name__ == '__main__':
    main(sys.argv[1:])
