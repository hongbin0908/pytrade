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

def get_lipt():
    lipt = ["date", "sym", "open", "high", "low", "close", 'volume','label5']
    with open("main/ta/ta_base1_importance") as fipt:
        idx = 0
        for each in fipt.readlines():
            tokens = each.split(",")
            lipt.append(tokens[0].strip())
            idx += 1
            if idx == 29:
                break
    return lipt

span = 7

def adx_signal(row):
    #if row["cross_%d" % span] > 1 and row["cross_%d_shift_1" % span] < 1 and row["ta_ADX_%d_%d" % (span, span)] > 25:
    if row["cross_%d" % span] > 0.8  and row['cross_%d'% span ] < 1.4 and row["ta_ADX_%d_%d" % (span, span)] > 30:
        return 1.0
    return 0.0
def adx_sig(df):
    highs = df['high'].values
    lows = df['low'].values
    closes=df['close'].values

    df = df.join(pta.MDI(df, span))
    df = df.join(pta.PDI(df, span))
    df = df.join(pta.ADX(df, span, span))
    
    df["cross_%d" % span] = df["ta_PDI_%d" % span] / df["ta_MDI_%d" % span]
    df["cross_%d_shift_1" % span ] = df["cross_%d" % span].shift(1)
    df["ta_adx_signal_%d" % span] = df.apply(lambda row: adx_signal(row), axis = 1) 
    del df["ta_MDI_%d" % span]
    del df["ta_PDI_%d" % span]
    del df["ta_ADX_%d_%d" % (span,span)]
    del df["cross_%d" % span] 
    del df["cross_%d_shift_1" % span]
    return df
def main(df):
    df = df.reset_index("date", drop = True)
    df = base1.main(df)
    lipt = get_lipt()
    df  =  df[lipt]
    del df["ta_ADX_7"]
    df = adx_sig(df)
    df = df.eset_index()
    assert 37 == df.shape[1]
    return df

