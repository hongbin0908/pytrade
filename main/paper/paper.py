#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

"""
$ ./paper/paper.py  tadowcall1_GBCv1n322md3lr001_l5_s1700e2009 call1_dow 2010-01-01 2016-12-31 2 0.62

151 211 0.715639810427
"""


import sys,os
import json
import numpy as np
import pandas as pd
from sklearn.externals import joblib # to dump model

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)
sys.path.append(local_path)

import main.base as base
import main.ta as ta


def get_df(taName):
    dfTa = ta.get_merged(os.path.join(root, 'data', 'ta', taName))
    return dfTa

def get_cls(clsName):
    cls = joblib.load(os.path.join(root, 'data', 'models',"model_" + clsName + ".pkl"))
    return cls
def select_(dfTa, top, thresh):
    dfTa = dfTa.loc[dfTa['pred'] >= thresh]
    dfTa = dfTa.sort_values(["date", "pred"],ascending = False)
    dfTa = dfTa.groupby('date').head(top)
    return dfTa

def pre_rank(df):
    df['rank'] = np.arange(len(df)) + 1
    return df

def accu(df, label):
    npLabel = df[label].values
    npTrue = npLabel[(npLabel>1.0)]
    print npTrue.size, npLabel.size, npTrue.size*1.0/npLabel.size

def get_range(df, start ,end):
    return df.query('date >="%s" & date <= "%s"' % (start, end))

def main(argv):
    clsName = argv[1]
    taName = argv[2]
    start = argv[3]
    end = argv[4]
    top = argv[5]
    thresh = float(argv[6])
    dfTa = get_df(taName)
    dfTa = get_range(dfTa, start, end)
    cls = get_cls(clsName)
    feat_names = base.get_feat_names(dfTa)
    npFeat = dfTa.loc[:,feat_names].values
    #for i, npPred in enumerate(cls.staged_predict_proba(npFeat)):
    #    if i == 322:
    #        break
    npPred = cls.predict_proba(npFeat)
    dfTa["pred"] = npPred[:,1]
    accu(select_(dfTa, int(top), thresh), "label5")
if __name__ == '__main__':
    main(sys.argv)
