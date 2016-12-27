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

def main(df):
    def call(row):
        if row['max'] == row['close'] and row['close'] > row['open']:
            return 1
        return 0
    df.reset_index(drop=True, inplace=True)
    df.loc[:,'max'] = df['close'].rolling(100).min()
    df.loc[:,'close_shift1'] = df["close"].shift(1)
    df.loc[:,'range'] = abs(df["close"] - df['open'])
    df.loc[:, 'range_max'] = df['range'].rolling(2).min()

    df["ta_sig_upbreak"] = df.apply(lambda row: call(row), axis = 1)

    del df["max"]
    del df["range"]
    del df["range_max"]
    del df["close_shift1"]
    return df
