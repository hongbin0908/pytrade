#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

import sys,os
import json
import numpy as np
import pandas as pd
import math
import multiprocessing
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

def get_file_list(rootdir, ext=".csv"):
    file_list = []
    for f in os.listdir(rootdir):
        if f == None or not f.endswith(ext):
            continue
        file_list.append(os.path.join(rootdir, f))
    return file_list

def get_stock_from_path(pathname):
    return os.path.splitext(os.path.split(pathname)[-1])[0]

def get_feat_names(df):
    """
    the the columns of features name to train
    """
    return [x for x in df.columns if x.startswith('ta_')]

def get_all_from(path,ext=".pkl"):
    sym2df = {}
    for each in get_file_list(path,ext=ext):
        symbol = get_stock_from_path(each)
        df = pd.read_pickle(each)
        df["sym"] = symbol
        sym2df[symbol] = df
    return sym2df
