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

from main.utils import time_me
import main.pandas_talib as pta
import main.base as base

import main.ta.ta_base1 as base1
import main.ta.ta_cdl as cdl

def merge(df, depth):
    dfStable = pd.read_pickle(os.path.join(dataroot,
                              "phase1_dump",
                              "sp500_base1_%d_stable.pkl" % depth))
    dfStable = dfStable[dfStable.direct != 0]
    tobe = []
    for i, each in  dfStable.iterrows():
        name = each["name"]
        fname = each["fname"]
        start = each["start"]
        end = each["end"]
        new = df.apply(lambda row:
                     1 if ((row[fname] >= start) and (row[fname] < end)) else 0, axis=1)
        print ".",
        tobe.append(pd.Series(new, name = name))
    print
    df = df.join(pd.DataFrame(tobe).transpose())
    return df
def main(df):
    df = base1.main(df)
    df.reset_index(inplace=True,drop=True)
    orig_feats = base.get_feat_names(df)
    print df.shape
    df = merge(df, 1)
    print df.shape
    df = merge(df, 2)
    print df.shape
    for each in orig_feats:
        del df[each]
    print df.shape
    return df

