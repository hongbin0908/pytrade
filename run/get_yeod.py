#!/usr/bin/env python3

import os, sys, time
import pandas as pd
import numpy as np
import pandas_datareader.yahoo.daily as yahoo
import multiprocessing
import subprocess
import logging

logging.basicConfig(level=logging.DEBUG)

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)


#cmdstr = """
#    cd {target};
#    rm -rf sp100_current.tar.gz;
#    wget http://hongindex.com/sp100_current.tar.gz;
#    mkdir -p yeod
#    rm -rf yeod/*
#    tar xvzf sp100_current.tar.gz -C yeod
#    #rsync  -av /data/users/hongbin/wp/pytrade hongbin@bc-r2hdp2:~/wp/
#    """.format(**{"target":os.path.join(local_path, "..", "data")})
#logging.debug(cmdstr)
#subprocess.check_output(cmdstr, shell=True)
import zipfile
import urllib.request
def get(snapname):
    dirname = "sp100_snapshot_%s" % snapname
    url = "http://hongindex.com/yeod/%s.zip" % dirname
    zfpath = os.path.join(root, "data", "yeod", "%s.zip" % dirname)
    urllib.request.urlretrieve(url, zfpath)
    zf = zipfile.ZipFile(zfpath)
    zf.extractall(os.path.join(root, 'data', 'yeod', dirname))

def get2(snapname):
    dirname = "sp500_snapshot_%s" % snapname
    url = "http://hongindex.com/yeod/%s.zip" % dirname
    zfpath = os.path.join(root, "data", "yeod", "%s.zip" % dirname)
    urllib.request.urlretrieve(url, zfpath)
    zf = zipfile.ZipFile(zfpath)
    zf.extractall(os.path.join(root, 'data', 'yeod', dirname))
if __name__  == '__main__':
    get("20091129")
    get("20100710")
    get("20140321")
    get("20161110")
    get("20120316")
    #get2("20091231")
