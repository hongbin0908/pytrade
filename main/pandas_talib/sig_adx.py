#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

import sys,os
import json
import numpy as np
import pandas as pd
from sklearn.externals import joblib # to dump model

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

import main.base as base
import main.ta as ta
import main.yeod.yeod as yeod
import main.ta.build as build
import main.pandas_talib as pta
import talib

span = 14

def adx_signal(row):
    #if row["cross_%d" % span] > 1 and row["cross_%d_shift_1" % span] < 1 and row["ta_ADX_%d_%d" % (span, span)] > 25:
    #if row["cross_%d" % span] < 0.45 and row["cross_%d" % span] <  row["cross_%d_shift_1" % span]  and\
    #    row["ta_ADX_%d_%d" % (span, span)] > 15:
    if row["cross_%d" % span] < 0.30  and row["cross_%d" % span] <  row["cross_%d_shift_1" % span]  and\
        row["ta_ADX_%d_%d" % (span, span)] > 20:
    #if row["cross_%d" % span] > 1  and row['cross_%d'% span ] < 1.4 and row["ta_ADX_%d_%d" % (span, span)] > 25:
        return 1.0
    return 0.0
def main(df):
    highs = df['high'].values
    lows = df['low'].values
    closes=df['close'].values

    df = df.join(pta.MDI(df, span))
    df = df.join(pta.PDI(df, span))
    df = df.join(pta.ADX(df, span, span))
    
    df["cross_%d" % span] = df["ta_PDI_%d" % span] / df["ta_MDI_%d" % span]
    df["cross_%d_shift_1" % span ] = df["cross_%d" % span].shift(1)
    df["ta_sig_adx"] = df.apply(lambda row: adx_signal(row), axis = 1) 
    del df["ta_MDI_%d" % span]
    del df["ta_PDI_%d" % span]
    del df["ta_ADX_%d_%d" % (span,span)]
    del df["cross_%d" % span] 
    del df["cross_%d_shift_1" % span]
    return df

