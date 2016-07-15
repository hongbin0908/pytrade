#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

"""
./pred/pred.py  call1s1_dow_GBCv1n322md3lr001_l5_s1700e2009 call1s1_dow 2016-01-01 2016-12-31  label5
"""

import sys,os
import json
import numpy as np
import pandas as pd
import math
import multiprocessing
import cPickle as pkl
from sklearn.externals import joblib # to dump model
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)
sys.path.append(local_path)

from main.utils import time_me
import main.base as base

def get_cls(clsName):
    return joblib.load(os.path.join(root, 'data', 'models', "model_" + clsName + ".pkl"))
def get_ta(taName):
    return base.get_merged_with_na(os.path.join(root, 'data', 'ta', taName))

def main(argv):
    clsName = argv[0]
    stage = int(argv[1])
    taName = argv[2]
    start = argv[3]
    end = argv[4]
    label = argv[5]

    if end == "<0":
        end = "2099-12-31"

    cls = get_cls(clsName)
    ta = get_ta(taName)
    ta = ta[ (ta.date >= start) & (ta.date<=end) ]

    dfFeat = ta.loc[:, base.get_feat_names(ta)]
    print dfFeat.tail(1)
    npFeat = dfFeat.values
    #npPred = cls.predict_proba(npFeat)
    for i, npPred in enumerate(cls.staged_predict_proba(npFeat)):
        if i == stage:
            break
    ta["pred"] = npPred[:,1]
    ta.sort("pred", inplace = True, ascending = False)
    print ta[["date","sym", "pred"]].head(10)
    ta.to_csv(os.path.join(base.dir_preds(), base.fname_pred(clsName, taName,start,end)))
    ta[["date", "sym", "pred", label]].to_csv(os.path.join(base.dir_preds(), base.fname_pred_s(clsName, taName, start, end)))

if __name__ == '__main__':
    main(sys.argv[1:])
