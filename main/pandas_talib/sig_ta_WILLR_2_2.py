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

import talib

span = 14

def adx_signal(row):
    if row["ta_tmp"] >= -90 and row['ta_tmp'] < -60:
        return 1.0
    return 0.0
def main(df):
    highs = df['high'].values
    lows = df['low'].values
    closes=df['close'].values

    df["ta_tmp"] = talib.WILLR(highs,lows,closes,2)
    
    df["ta_sig_ta_WILLR_2_2"] = df.apply(lambda row: adx_signal(row), axis = 1) 
    del df["ta_tmp"]
    return df
