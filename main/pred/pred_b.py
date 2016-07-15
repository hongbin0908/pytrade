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
    model_father = os.path.join(root, 'data', 'models_batch', clsName)
    for d in sorted(os.listdir(model_father)):
        print d
        cls = joblib.load(os.path.join(model_father, d, "model.pkl"))
        ta = base.get_merged_with_na(os.path.join(root, 'data', 'ta_batch', taName, d))

        dfFeat = ta.loc[:, base.get_feat_names(ta)]
        dfFeat = dfFeat[(dfFeat['date'] >= start) & (dfFeat['date'] <= end)]
        npFeat = dfFeat.values
        #npPred = cls.predict_proba(npFeat)
        for i, npPred in enumerate(cls.staged_predict_proba(npFeat)):
            if i == stage:
                break
        ta["pred"] = npPred[:,1]
        ta.sort("pred", inplace = True, ascending = False)
        print ta[["date","sym", "pred"]].head(10)
        out_dir = os.path.join(root, 'data', 'pred_batch', "%s_%s_%s_%s_%s_%s" % (clsName,stage,taName,start,end,label))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        ta.to_csv(os.path.join(out_dir, "pred.csv"))
        ta[["date", "sym", "pred", label]].to_csv(os.path.join(out_dir, 'pred.s.csv'))

if __name__ == '__main__':
    main(sys.argv[1:])
