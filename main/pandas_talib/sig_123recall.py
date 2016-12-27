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

def main(df):
    def call(row):
        if row["adx"] > 30 \
           and row["diff"] < 1.0 \
           and row["diff_shift1"] < 1.0 \
           and row["diff_shift2"] < 1.005 :
            return 1
        return 0
    df.reset_index(drop=True, inplace=True)
    df["adx"] = pta.ADX(df, 14, 7)
    df["diff"] = df.close / df.close.shift(1)
    
    df["diff_shift1"] = df["diff"].shift(1)
    df["diff_shift2"] = df["diff"].shift(2)
    df["diff_shift3"] = df["diff"].shift(3)

    df["ta_sig_123recall"] = df.apply(lambda row: call(row), axis = 1)

    del df["adx"]
    del df["diff"]
    del df["diff_shift1"]
    del df["diff_shift2"]
    del df["diff_shift3"]
    return df
