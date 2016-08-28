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
        if np.isnan(row['IvCall10']):
            return -1
        if row['IvCall10'] > row["IvCall10_shift1"] * 1.0: # and row['IvPut10'] < row['IvPut10_shift1'] *1.0 and \
            return 1
        return 0
    df.reset_index(drop=True, inplace=True)
    sym = df.ix[0].loc['sym']
    df_vol = pd.read_csv(os.path.join(base.dir_vol(), sym + ".csv"))
    d = {'Date':'date'}
    df_vol = df_vol.rename(columns=d)
    df_vol = df_vol[["date", "IvCall10", "IvPut10"]]
    df = pd.merge(how = 'left', left = df, right =df_vol, left_on = 'date', right_on = 'date')

    df.loc[:,"IvCall10_shift1"] = df["IvCall10"].shift(1)
    df.loc[:,"IvPut10_shift1"] = df["IvPut10"].shift(1)
    df.loc[:,"volume_shift1"] = df["volume"].shift(1)

    df.loc[:,"ta_sig_vol"] = df.apply(lambda row : call(row), axis = 1)

    del df["IvCall10"]
    del df["IvCall10_shift1"]
    del df["IvPut10"]
    del df["IvPut10_shift1"]
    del df["volume_shift1"]


    return df
