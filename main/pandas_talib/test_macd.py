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

fast = 12
slow = 26

def adx_signal(row):
    #if row["cross_%d_%d" % (fast,slow)] > 1 \
    #        and row["cross_%d_%dshift_1" % (fast,slow)]  < 1.1:
    #if row["cross_%d_%d" % (fast,slow)] < 0.33 \
    #        and row["cross_%d_%d" % (fast,slow)] < row["cross_%d_%dshift_1" % (fast,slow)]  :
    #if row["cross_%d_%d" % (fast,slow)] > 1 \
    #        and row["cross_%d_%dshift_1" % (fast,slow)]  < 1.1\
    #        and row["ta_MACD_%d_%d" %(fast,slow)] >0 and row["ta_MACD_%d_%d_shift_1"%(fast, slow)] < 0:
    if row["ta_MACD_%d_%d" %(fast,slow)] <0  and row["ta_MACD_%d_%d" %(fast,slow)] > -0.1 and row["ta_MACD_%d_%d" %(fast,slow)] > row["ta_MACD_%d_%d_shift_1"%(fast, slow)] :
        return 1.0
    return 0.0
def sig_macd(df):
    highs = df['high'].values
    lows = df['low'].values
    closes=df['close'].values

    df['ta_NATR_%d' % 14] = talib.NATR(highs, lows, closes, 14)
    df = df.join(pta.MACD(df, fast, slow))
    
    df["cross_%d_%d" % (fast, slow) ] = df["ta_MACD_%d_%d" % (fast, slow)] / df["ta_MACDsign_%d_%d" % (fast,slow)]
    df["ta_MACD_%d_%d_shift_1"%(fast,slow)] = df["ta_MACD_%d_%d"%(fast, slow)].shift(1)
    df["cross_%d_%dshift_1" % (fast, slow) ] = df["cross_%d_%d" % (fast, slow)].shift(1)
    df["ta_macd_signal_%d_%d" % (fast, slow)] = df.apply(lambda row: adx_signal(row), axis = 1) 
    return df
def main(args):
    build.work2(10, "sp500Top50", sig_macd)
    dfTa = base.get_merged("sig_macd", yeod.get_sp500Top50())
    dfTa1 = dfTa[(dfTa["date"] < "2010-01-01")]
    #dfTa = dfTa[(dfTa["date"] >= "2010-01-01") & (dfTa["date"]<="2010-12-31")]
    #dfTa = dfTa[(dfTa["date"]>"2010-12-31")]
    dfTa2 = dfTa1[dfTa1["ta_macd_signal_%d_%d" % (fast, slow)]>0]
    print len(dfTa2[dfTa2["label5"]>1.0])*1.0/len(dfTa2),
    print len(dfTa2[dfTa2["label5"]>1.0])*1.0,
    print len(dfTa1[dfTa1["label5"]>1.0])*1.0/len(dfTa1)

    dfTa1 = dfTa[(dfTa["date"] > "2010-01-01")]
    dfTa2 = dfTa1[dfTa1["ta_macd_signal_%d_%d" % (fast, slow)]>0]
    print len(dfTa2[dfTa2["label5"]>1.0])*1.0/len(dfTa2),
    print len(dfTa2[dfTa2["label5"]>1.0])*1.0,
    print len(dfTa1[dfTa1["label5"]>1.0])*1.0/len(dfTa1)
if __name__ == '__main__':
    main(sys.argv)

