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

dataroot = os.path.join(root, "data")

from main.utils import time_me
import main.pandas_talib as pta
import main.base as base

import main.ta.ta_base1 as base1
import main.ta.ta_cdl as cdl

def merge(df, depth):
    dfStable = pd.read_pickle(os.path.join(dataroot,
                              "model/meta_base1_sp500Top50_score5_1984-01-01_2009-12-31_%d_100000.pkl" % depth))
    dfStable = dfStable[dfStable.direct != 0]
    tobe = []
    for i, each in  dfStable.iterrows():
        name = each["name"]
        fname = each["fname"]
        start = each["start"]
        end = each["end"]
        new = df.apply(lambda row:
                     1 if ((row[fname] >= start) and (row[fname] < end)) else 0, axis=1)
        tobe.append(pd.Series(new, name = name))
    df = df.join(pd.DataFrame(tobe).transpose())
    return df
def main(df):
    df = base1.main(df)
    df.reset_index(inplace=True,drop=True)
    orig_feats = base.get_feat_names(df)
    df = merge(df, 1)
    df = merge(df, 2)
    df = merge(df, 3)
    for each in orig_feats:
        if not each.startswith("ta_ADX"):
            del df[each]
    print list(df.columns)
    return df

