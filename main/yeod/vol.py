#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

import sys,os
import json
import numpy as np
import pandas as pd
import urllib2
import math
import random
import multiprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib # to dump model
from sklearn import preprocessing

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

import main.model.model_param_set as params_set
import main.base as base
import main.ta as ta
import main.model.sample_weight as sw
import main.yeod.yeod_b as yeod_b

def download_vol(sym):
    url = "https://www.quandl.com/api/v3/datasets/VOL/%s.csv?api_key=-AsXbfQt8Nx2AMfUsEws" % sym
    print url
    f = urllib2.urlopen(url)
    buff = f.read()
    f.close()
    while not buff[0].isalnum():
        buff = buff[1:]
    return buff
def main(args):
    lsym = yeod_b.get_sp500Top100()
    dir_vol = base.dir_vol()

    for sym in lsym:
        file_vol_per_sym = os.path.join(dir_vol, sym + ".csv")
        if os.path.isfile(file_vol_per_sym):
            continue
        vol = download_vol(sym)
        with open(file_vol_per_sym, "w") as f:
            print >> f, vol


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='download vol')
    
    parser.add_argument('eod', help="the path of eod")

    args = parser.parse_args()
    main(args)
