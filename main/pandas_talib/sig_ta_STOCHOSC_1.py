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

import main.pandas_talib as pta

span = 14

def adx_signal(row):
    if row["ta_tmp"] >= 80:
        return 1.0
    if row['ta_tmp'] <= 20:
        return -1.0
    return 0.0
def main(df):
    highs = df['high'].values
    lows = df['low'].values
    closes=df['close'].values

    df["ta_tmp"] = pta.STOCHOSC(df,14)
    
    df["ta_sig_ta_STOCHOSC_1"] = df.apply(lambda row: adx_signal(row), axis = 1) 
    del df["ta_tmp"]
    return df
