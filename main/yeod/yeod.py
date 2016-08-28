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
import main.base as base

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

def get_sp500Top5():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort_values("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i, each in df.head(5).iterrows()]
def get_sp500Top50():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort_values("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i, each in df.head(50).iterrows()]
def get_sp500Top50p():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort_values("Market Cap", ascending=False)
    re = [each["Symbol"].strip() for i,each in df.head(50).iterrows()]
    re.append('^DJI')
    return re
def get_sp500Top100():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort_values("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i,each in df.head(100).iterrows()]

@time_me
def main(args):
    symbols = eval("get_%s" % args.setname)()
    engine.work(symbols, base.dir_eod(), args.poolnum)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="download eod")
    parser.add_argument('-p', '--pool', help="thread pool num", dest="poolnum", action="store", default=1, type=int)
    parser.add_argument('setname', help = "the sym set to be download")
    args = parser.parse_args()
    main(args)
