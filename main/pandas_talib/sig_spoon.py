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

def main(df):
    def call(row):
        if row["adx"] > 40 and row["is_in"] > 0 and row["is_in_shift1"] > 0:
            return 1.0
        return 0.0
    def call_in(row):
        if row['high'] > row['high_shift1'] and row['low'] < row['low_shift1']:
            return 1
        return 0
    df.reset_index(drop=True, inplace=True)
    df["high_shift1"] = df["high"].shift(1)
    df["low_shift1"] = df["low"].shift(1)
    df["is_in"] = df.apply(lambda row: call_in(row), axis=1)
    df["is_in_shift1"] = df["low"].shift(1)
    df["adx"] = pta.ADX(df, 14, 14)


    df["ta_sig_spoon"] = df.apply(lambda row: call(row), axis = 1)
    return df
