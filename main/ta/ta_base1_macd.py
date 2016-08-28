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
import main.ta.ta_cdl as cdl

def main(df):
    df['closer'] = df['close'] / df['close'].shift(1)
    closes=df['closer'].values
    df = base1.main(df)
    for tri in [(6,13,8),(12, 26, 9), (24, 52, 18)]:
        macd = talib.MACD(closes, tri[0], tri[1], tri[2])
        df['ta_MACD_macd_%d_%d_%d'%(tri[0],tri[1],tri[2])] = macd[0]
        df['ta_MACD_macdsignal_%d_%d_%d'%(tri[0],tri[1],tri[2])] = macd[1]
        df['ta_MACD_macdhist_%d_%d_%d'%(tri[0],tri[1],tri[2])] = macd[2]
    print df.shape
    return df

