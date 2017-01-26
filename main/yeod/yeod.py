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


class YeodBase():

    def get_name(self):
        pass

    def get_syms(self):
        pass


class Sp500(YeodBase):
    def __init__(self, idx, window = 100):
        self.idx = idx
        self.window = window
    def get_name(self):
        return "sp500w%di%d" % (self.window, self.idx)
    def get_syms(self):
        df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
        df = df.sort_values("Market Cap", ascending=False)
        list1 =  [each["Symbol"].strip() \
                for i, each in df.head(self.window*self.idx+self.window)\
                                .tail(self.window).iterrows()]
        if self.window > 250:
            list2 = [each.strip() for each in open(os.path.join(root, "sp500_deleted.csv")).readlines()]
        else:
            list2 = []
        list1.extend(list2)
        return list1


class Dow30(YeodBase):
    """
    dow jones symbols set.
    """
    def __init__(self, idx, window = 30):
        self.idx = idx
        self.window = window
    def get_name(self):
        return "Dow30w%di%d" % (self.window, self.idx)
    def get_syms(self):
        df = pd.read_csv(os.path.join(root, "dow30_2016.csv"))
        print(df)
        s = [each["Symbol"].strip() for i, each in df.head(self.window*self.idx+self.window).tail(self.window).iterrows()]
        return s

class sp100(YeodBase):
    """
    sp100 symbols set.
    """
    def __init__(self, idx, window = 30):
        self.idx = idx
        self.window = window
    def get_name(self):
        return "sp100w%di%d" % (self.window, self.idx)
    def get_syms(self):
        df = pd.read_csv(os.path.join(root, "sp100_all_2016.csv"))
        s = [each["Symbol"].strip() for i, each in df.head(self.window*self.idx+self.window).tail(self.window).iterrows()]
        return s

class TestSyms(YeodBase):
    """
    only for test
    """
    def __init__(self, idx, window = 2):
        self.idx = idx
        self.window = window
    def get_name(self):
        return "testw%di%d" % (self.window, self.idx)
    def get_syms(self):
        return ["MSFT", "AAPL", "MMM"]

def get_test_list(window):
    return [TestSyms(i, window) for i in range(int(2/window))]
def get_sp500_list(window):
    return [ Sp500(i, window) for i in range(int(500/window))]

def get_dow30_list(window):
    return [ Dow30(i, window) for i in range(int(30/window))]

def get_sp100_list(window):
    return [ sp100(0, 150)]


def get_sp500Top50():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort_values("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i, each in df.head(50).iterrows()]


get_sp100_list(10)
# --------------------------------------------------------------------------

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


def get_sp500R50(i, window = 50):
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort_values("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i, each in df.head(window*i+window).tail(window).iterrows()]


def get_sp500():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort_values("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i, each in df.iterrows()]
def get_sp500Rng100():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort_values("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i, each in df.head(100).tail(50).iterrows()]
def get_sp500Top5():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort_values("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i, each in df.head(5).iterrows()]
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

def main2(poolnum=10, target=base.dir_eod(), symbols = sp100(0, 150).get_syms()):
    engine.work(list(set(symbols)), target, poolnum)

def main(args):
    main2(args.poolnum)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="download eod")
    parser.add_argument('-p', '--pool', help="thread pool num", dest="poolnum", action="store", default=1, type=int)
    parser.add_argument('setname', help = "the sym set to be download")
    args = parser.parse_args()
    main(args)

def test_main2():
    import tempfile
    tmpdir = tempfile.TemporaryDirectory()
    main2(1, tmpdir.name, ["AAPL", "YHOO"])
    import glob
    assert 2 == len(glob.glob(tmpdir.name + "/*.csv"))
