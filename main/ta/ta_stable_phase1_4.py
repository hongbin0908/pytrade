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

dataroot = os.path.join(root, "data", "feat_select")

import main.pandas_talib as pta
import main.base as base

import main.ta.ta_base1 as base1
import main.ta.ta_stable_phase1_3 as stable_3
from main.model import feat_select

def main(df):
    df1 = base1.main(df)
    df1.reset_index(drop=True, inplace=True)

    df2 = stable_3.main(df)
    df2.reset_index(drop=True, inplace=True)

    l = ['ta_CMO_14', 'ta_RSI_14','ta_CMO_7','ta_RSI_7','ta_CMO_10',
         'ta_RSI_10',
         'ta_TRIX_2',
         'ta_RSI_28',
         'ta_CMO_28',
         'ta_RSI_5',
         'ta_STOCHRSI_slowd_5_20_12',
         'ta_ROC_7',
         'ta_ROCR100_7',
         'ta_ROCP_7',
         'ta_ROCR_7',
         'ta_STOCHRSI_slowd_7_20_12',
         'ta_RSI_2',
         'ta_ROC_5',
         'ta_ROCR100_5',
         'ta_ROCR_5',
         'ta_ROCP_5',
         'ta_WILLR_10',
         'ta_WILLR_14',
         'ta_ROC_2',
         'ta_ROCR100_2',
         'ta_ROCP_2',
         'ta_ROCR_2',
         'ta_WILLR_7']
    df1 = feat_select.append_deep_feats(df1,l)

    df1 = df1[base.get_feat_names(df1).extend(["date"])]
    print df2.shape
    df2 = df2.merge(df1, left_on='date', right_on="date", how='inner')
    print df2.shape
    return df2

