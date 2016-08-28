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
        if row["is_do_25_shift2"] > 0 and row["is_up_25_shift1"] > 0 \
                and row["close"] > row["close_shift1"] \
                and row["close"] > row["ma10"] \
                and row["close"] > row["ma50"]:
            return 1.0
        return 0.0
    def call_do_25(row):
        if row['high'] - row['low'] == 0:
            return 0
        if (row["close"]- row["low"]) / (row["high"]-row['low']) < 0.3:
            return 1
        return 0
    def call_up_25(row):
        if row['high'] - row['low'] == 0:
            return 0
        if (row["high"]- row["close"]) / (row["high"]-row['low']) < 0.3:
            return 1
        return 0
    df.reset_index(drop=True, inplace=True)

    df["is_do_25"] = df.apply(lambda row : call_do_25(row), axis=1)
    df["is_do_25_shift2"] = df["is_do_25"].shift(2)
    df["is_up_25"] = df.apply(lambda row : call_up_25(row), axis=1)
    df["is_up_25_shift1"] = df["is_up_25"].shift(1)

    df["ma10"] = pta.MA(df, 10)
    df["ma50"] = pta.MA(df, 50)

    df["close_shift1"] = df["close"].shift(1)

    df["ta_sig_180du"] = df.apply(lambda row: call(row), axis = 1)
    return df
