#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author hongbin@youzan.com

import os,sys
import talib
import numpy as np
import pandas as pd

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

from main.utils import time_me
import main.pandas_talib as pta
import main.base as base

import main.ta.ta_base1 as base1
import main.pandas_talib as pta

def get_lipt():
    lipt = ["date", "sym", "open", "high", "low", "close", 'volume', 'label5']
    with open("main/ta/ta_base1_importance") as fipt:
        idx = 0
        for each in fipt.readlines():
            tokens = each.split(",")
            lipt.append(tokens[0].strip())
            idx += 1
            if idx == 29:
                break
    return lipt

def main(df):
    df = base1.main(df)
    lipt = get_lipt()
    df  =  df[lipt]

    df.reset_index(drop = True)
    
    print df.shape
    for i in range(0, 14):
        df["ta_blank_1_%d" % i] = 1 
    for i in range(14, 28):
        df["ta_blank_2_%d" % i] = 1
    print df.shape
    #for i in (28,):
    #    ma = pta.MA(df, i)
    #    df["ta_MA_diff_%d" % i] = 100 * df["close"] / ma
    #for d in [(5, 7), (7,14), (14,21), (21,28), (7,21), (7,28)]:
    #    df["ta_MA_diff_%d_%d"%(d[0],d[1])] = 100 * pta.MA(df,d[1])/pta.MA(df,d[0])
    return df
