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
import main.pandas_talib.sig_123recall as sig_123recall
import main.pandas_talib.sig_adx as sig_adx
import main.pandas_talib.sig_upbreak as sig_upbreak
import main.pandas_talib.sig_ta_PLUS_DM_28 as sig_ta_PLUS_DM_28
import main.pandas_talib.sig_ta_WILLR_2_1 as sig_ta_WILLR_2_1
import main.pandas_talib.sig_ta_WILLR_2_2 as sig_ta_WILLR_2_2
import main.pandas_talib.sig_ta_STOCHOSC_1 as sig_ta_STOCHOSC_1

def get_name():
    return "ta_base1_ext5"
def main(df):
    df = df.reset_index("date", drop = True)
    df = base1.main(df)
    df = sig_123recall.main(df)
    df = sig_adx.main(df)
    df = sig_upbreak.main(df)
    df = sig_ta_PLUS_DM_28.main(df)
    df = sig_ta_WILLR_2_1.main(df)
    df = sig_ta_STOCHOSC_1.main(df)
    #df = sig_ta_WILLR_2_2.main(df)
    del df["ta_PLUS_DM_28"]
    #del df["ta_WILLR_2"]

    #assert 40 == df.shape[1]
    return df
