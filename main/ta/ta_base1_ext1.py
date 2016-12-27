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
import main.ta.ta_base1_top29 as top29
import main.pandas_talib.sig_123recall as sig_123recall
import main.pandas_talib.sig_adx as sig_adx
import main.pandas_talib.sig_upbreak as sig_upbreak

"""
add history value of
ta_ROC_5
ta_WILLR_2
ta_ROC_7
ta_WILLR_7
ta_WILLR_5
ta_STOCHRSI_slowd_28_5_3
ta_NATR_28
"""


def main(df):
    df = df.reset_index("date", drop = True)
    df = base1.main(df)
    for each in ["ta_ROC_5","ta_WILLR_2", "ta_ROC_7", "ta_WILLR_7", "ta_WILLR_5", "ta_STOCHRSI_slowd_28_5_3", "ta_NATR_28"]:
        for i in [1, 2, 5, 7, 14]:
            df["%s-shift-%d" % (each, i)] = df[each].shift(i)
    return df

