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
    lipt = ["date", "sym", "open", "high", "low", "close", 'volume']
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
    sym = df['sym'].values[0]
    vol = pd.read_csv(os.path.join(base.dir_vol(), sym + ".csv"))
    d = {'Date':'date'}
    for each in vol.columns[1:]:
        d[each] = 'ta_' + each
    vol = vol.rename(columns=d)
    vol = vol.fillna(0)
    dfDate = df[["date"]]
    vol = pd.merge(how = 'left', left = dfDate, right = vol, left_on = 'date', right_on = 'date')
    vol = vol.fillna(0)
    print df.shape[1]
    df = pd.merge(how = 'left', left = df, right = vol, left_on = "date", right_on = "date")
    print df.shape[1]
    return df
