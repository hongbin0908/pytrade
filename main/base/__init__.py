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
import datetime as dt
import time
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
def get_sym_from_path(pathname):
    return os.path.splitext(os.path.split(pathname)[-1])[0]

def get_feat_names(df):
    """
    the the columns of feature names to train
    """
    return [x for x in df.columns if x.startswith('ta_')]

def get_all(taname, lsym):
    sym2df = {}
    for sym in lsym:
        df = pd.read_pickle(os.path.join(dir_ta(taname), sym + ".pkl"))
        df["sym"] = sym
        sym2df[sym] = df
    return sym2df

def get_range(df, start ,end):
    """
    get the date between start(include) and end(*include*)
    """
    return df[(df.date>=start) & (df.date<=end)]

def merge(sym2feats):
    dfMerged = None
    toAppends = []
    for sym in sym2feats.keys():
        df = sym2feats[sym]
        if dfMerged is None:
            dfMerged = df
        else:
            toAppends.append(df)
    # batch merge speeds up!
    if len(toAppends) > 0:
        dfMerged =  dfMerged.append(toAppends)
    return dfMerged

def get_merged_with_na(taname, lsym):
    sym2ta = get_all(taname, lsym)
    df = merge(sym2ta)
    if df is None:
        return None
    if len(df) > 0 and "ta_NATR_14" in df:
        df = df[df['ta_NATR_14']>1.0]
    return df


def get_merged(taname, lsym):
    df = get_merged_with_na(taname, lsym)
    if df is None:
        return df
    df = df.replace([np.inf,-np.inf],np.nan)\
        .dropna()
    return df


def dir_eod():
    p = os.path.join(root, 'data', 'yeod')
    if not os.path.exists(p):
        os.makedirs(p)
    return p
def dir_vol():
    p = os.path.join(root, 'data', 'vol')
    if not os.path.exists(p):
        os.makedirs(p)
    return p
def dir_ta(taname):
    p = os.path.join(root, 'data', 'ta', taname)
    if not os.path.exists(p):
        os.makedirs(p)
    return p
def dir_model():
    p = os.path.join(root, 'data', 'model')
    if not os.path.exists(p):
        os.makedirs(p)
    return p
def file_model(args):
    fname = "%s-%s-%s-%s-%s-%s" % (args.setname, args.taname, \
            args.clsname, args.labelname, args.start, args.end)
    if args.sw:
        fname += "-%s" % args.sw
    if args.repeat:
        fname += "-%s" % args.repeat
    if args.sample:
        fname += "-%s" % args.sample
    return (os.path.join(dir_model(), fname + ".pkl"), os.path.join(dir_model(), fname + ".ipt"))

def dir_paper():
    p = os.path.join(root, 'data', 'paper')
    if not os.path.exists(p):
        os.makedirs(p)
    return p
def file_paper(args):
    fname="%s-%s-%s-%s-%s-%d-%d" % (args.model,\
            args.setname,
            args.stage,
            args.start,
            args.end,
            args.top,
            args.thresh)
    return (os.path.join(dir_paper(), fname + ".report"), os.path.join(dir_paper(), fname + ".pre.csv"))

def dir_pred():
    p = os.path.join(root, 'data', 'pred')
    if not os.path.exists(p):
        os.makedirs(p)
    return p

def file_pred(args):
    fname = "%s-%s-%s-%s-%s-%s-%d" % (
            args.model,
            args.setname,
            args.taname,
            args.label,
            args.start,
            args.end,
            args.stage
            )
    return (os.path.join(dir_pred(), fname + ".report"),\
            os.path.join(dir_pred(), fname + ".csv")\
            )




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

def yeod(field, sym):
    p = os.path.join(root, 'data', 'yeod', field, sym + ".csv")
    return pd.read_csv(p)

def ta(field, sym):
    p = os.path.join(root, 'data', 'ta', field, sym + ".pkl")
    return pd.read_pickle(p)

def fname_pred(cls, ta, start, end):
    return cls + "_" + ta + "_" + start + "_"+end + ".csv"
def fname_pred_s(cls, ta, start, end):
    return cls + "_" + ta + "_" + start + "_"+end + ".s.csv"

def last_trade_date():
    """
    get the last trade date
    """
    df = pd.read_csv(os.path.join(dir_eod(), '^DJI.csv'))
    return df.date.max()


def strDate2num(str):
    df =dt.datetime.strptime(str, "%Y-%m-%d")
    return df
    #time_sec_float = time.mktime(df.timetuple())
    #return time_sec_float
