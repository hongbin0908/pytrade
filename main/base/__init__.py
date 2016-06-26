#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

"""
the base method use by pytrade
"""

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
    the the columns of feature names to train
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

def get_range(df, start ,end):
    """
    get the date between start(include) and end(*include*)
    """
    return df[(df.date>=start) & (df.date<=end)]


def get_merged_with_na(ta):
    sym2ta = base.get_all_from(ta)
    df = merge(sym2ta)
    df = df[df['ta_NATR_14']>1.0]
    return df


def get_merged(ta):
    df = get_merged_with_na(ta)
    df = df.replace([np.inf,-np.inf],np.nan)\
        .dropna()
    return df


def dir_preds():
    p = os.path.join(root, 'data','preds')
    if not os.path.exists(p):
        os.makedirs(p)
    return p

def dir_models():
    p = os.path.join(root, 'data', 'models')
    if not os.path.exists(p):
        os.makedirs(p)
    return p

def dir_yeod_dow():
    p = os.path.join(root, 'data', 'yeod', 'dow')
    if not os.path.exists(p):
        os.makedirs(p)
    return p

def yeod_dow_sym(sym):
    p = os.path.join(dir_yeod_dow(), sym + ".csv")
    return pd.read_csv(p)

def fname_pred(cls, ta, start, end):
    return cls + "_" + ta + "_" + start + "_"+end + ".csv"
def fname_pred_s(cls, ta, start, end):
    return cls + "_" + ta + "_" + start + "_"+end + ".s.csv"

def last_trade_date():
    """
    get the last trade date
    """
    df = yeod_dow_sym('MSFT')
    return df.date.max()
