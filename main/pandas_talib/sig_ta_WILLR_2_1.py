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
    if row["w2"] < -80:
        return 1.0
    return 0.0
def main(df):
    highs = df['high'].values
    lows = df['low'].values
    closes=df['close'].values
    dftmp = df[[]]
    dftmp["w"] = talib.WILLR(highs,lows,closes,50)
    dftmp["w2"] = dftmp["w"].shift(1)
    
    df["ta_sig_ta_WILLR_2_1"] = dftmp.apply(lambda row: adx_signal(row), axis = 1) 
    return df
