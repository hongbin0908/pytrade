#!/usr/bin/env python3

import os, sys, time
import pandas as pd
import numpy as np
import pandas_datareader.yahoo.daily as yahoo
import multiprocessing
import subprocess
import logging
import zipfile
import urllib.request

logging.basicConfig(level=logging.DEBUG)

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

from main import base


def get(snapname):
    dirname = "sp100_snapshot_%s" % snapname
    url = "http://www.hongindex.com/yeod/%s.zip" % dirname
    zfpath = os.path.join(root, "data", "yeod", "%s.zip" % dirname)
    urllib.request.urlretrieve(url, zfpath)
    zf = zipfile.ZipFile(zfpath)
    zf.extractall(os.path.join(root, 'data', 'yeod', dirname))

def get2(snapname):
    last_trade_date = base.get_last_trade_date()
    #last_trade_date = '2017-05-22'
    dirname = "sp500_snapshot_%s_%s" % (snapname, last_trade_date)
    url = "http://www.hongindex.com/yeod/%s.zip" % dirname
    zfdir = os.path.join(root, "data", "yeod")
    if not os.path.exists(zfdir):
        os.makedirs(zfdir)
    zfpath = os.path.join(root, "data", "yeod", "%s.zip" % dirname)
    urllib.request.urlretrieve(url, zfpath)
    zf = zipfile.ZipFile(zfpath)
    zf.extractall(os.path.join(root, 'data', 'yeod', dirname))
def get_index():
    dirname = "index"
    url = "http://www.hongindex.com/yeod/index.zip"
    zfpath = os.path.join(root, "data", "yeod", "%s.zip" % dirname)
    urllib.request.urlretrieve(url, zfpath)
    zf = zipfile.ZipFile(zfpath)
    zf.extractall(os.path.join(root, 'data', 'yeod', dirname))

if __name__  == '__main__':
    #get("20091129")
    #get("20100710")
    #get("20140321")
    #get("20161110")
    #get("20120316")
    get2("20091231")
    #get2("20101207")
    #get2("20111231")
    #get2("20121229")
    #get2("20131229")
    #get2("20141229")
    #get2("20151228")
    #get_index()
