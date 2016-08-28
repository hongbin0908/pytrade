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

def main(df):
    df = base1.main(df)
    lipt = get_lipt()
    df  =  df[lipt]
    df["shift"] =  df['close'].shift(5)
    df["trend"] = df['close'] / df['shift']
    df = df[df["trend"] < 1.0]
    del df["trend"]
    del df["shift"]
    assert 37 == df.shape[1]
    return df

