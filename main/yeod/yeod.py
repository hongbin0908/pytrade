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


class sp100_snapshot(YeodBase):
    """
    """
    def __init__(self, snap):
        self.snap = snap
    def get_name(self):
        return "sp100_snapshot_%s" % (self.snap)
    def get_syms(self):
        df = pd.read_csv(os.path.join(root, "sp100_snapshot", "sp100_%s.CSV" % self.snap))
        s = [each["Symbol"].strip() for i, each in df.iterrows()]
        return s

class sp500_snapshot(YeodBase):
    """
    """
    def __init__(self, snap):
        self.snap = snap
    def get_name(self):
        return "sp500_snapshot_%s" % (self.snap)
    def get_syms(self):
        df = pd.read_csv(os.path.join(root, "sp100_snapshot", "sp500_%s.CSV" % self.snap))
        s = [each["Symbol"].strip() for i, each in df.iterrows()]
        return s
class TestSyms(YeodBase):
    """
    only for test
    """
    def __init__(self):
        pass
    def get_name(self):
        return "test" 
    def get_dir_name(self):
        return "test"
    def get_syms(self):
        return ["MSFT", "AAPL", "MMM"]

def get_test_list():
    return TestSyms()

def get_sp100_snapshot_20081201():
    return sp100_snapshot("20081201")
def get_sp100_snapshot_20091129():
    return sp100_snapshot("20091129")
def get_sp100_snapshot_20100710():
    return sp100_snapshot("20100710")
def get_sp100_snapshot_20120316():
    return sp100_snapshot("20120316")
def get_sp100_snapshot_20140321():
    return sp100_snapshot("20140321")
def get_sp100_snapshot_20151030():
    return sp100_snapshot("20151030")
def get_sp100_snapshot_20161110():
    return sp100_snapshot("20161110")
def get_sp500_snapshot_20091231():
    return sp500_snapshot("20091231")


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

