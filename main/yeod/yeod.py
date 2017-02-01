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

class sp100_snapshot(YeodBase):
    """
    """
    def __init__(self, snap, idx = 0, window = 150):
        self.snap = snap
        self.idx = idx
        self.window = window
    def get_name(self):
        return "sp100snap%sw%di%d" % (self.snap, self.window, self.idx)
    def get_dir_name(self):
        return "sp100_snapshot_%s" % self.snap
    def get_syms(self):
        df = pd.read_csv(os.path.join(root, "sp100_snapshot", "sp100_%s.CSV" % self.snap))
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
    def get_dir_name(self):
        return "test"
    def get_syms(self):
        return ["MSFT", "AAPL", "MMM"]

def get_test_list(window):
    return [TestSyms(i, window) for i in range(int(150/window))]
def get_sp500_list(window):
    return [ Sp500(i, window) for i in range(int(500/window))]

def get_dow30_list(window):
    return [ Dow30(i, window) for i in range(int(30/window))]


def get_sp100_snapshot_20081201(window):
    return[sp100_snapshot("20081201")]
def get_sp100_snapshot_20091129(window):
    return [ sp100_snapshot("20091129")]
def get_sp100_snapshot_20100710(window):
    return [ sp100_snapshot("20100710")]
def get_sp100_snapshot_20120316(window):
    return [ sp100_snapshot("20120316")]
def get_sp100_snapshot_20140321(window):
    return [ sp100_snapshot("20140321")]
def get_sp100_snapshot_20151030(window):
    return [ sp100_snapshot("20151030")]
def get_sp100_snapshot_20161110(window):
    return [ sp100_snapshot("20161110")]


def get_sp500Top50():
    df = pd.read_csv(os.path.join(root, "constituents-financials.csv"))
    df = df.sort_values("Market Cap", ascending=False)
    return [each["Symbol"].strip() for i, each in df.head(50).iterrows()]

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

def main2( poolnum, target, symbols):
    import zipfile
    zf = zipfile.ZipFile(target, mode='w')

    import tempfile
    tmpdir = tempfile.TemporaryDirectory().name
    print(tmpdir)
    engine.work(list(set(symbols)), tmpdir, poolnum)
    contents = os.walk(tmpdir)
    for root, folders, files in contents:
        for file_name in files:
            zf.write(os.path.join(root, file_name), file_name)
    zf.close()


if __name__ == '__main__':
    if base.is_test_flag():
        main2(poolnum=1, 
                target=os.path.join(root, "main", "yeod", "yeod_demo.zip"),
                symbols = sp100_snapshot("20091129").get_syms()[0:10])

