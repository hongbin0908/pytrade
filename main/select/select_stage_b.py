#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os
import json
import time
import numpy as np
import pandas as pd
from sklearn.externals import joblib # to dump model
import cPickle as pkl
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)
sys.path.append(local_path)

import main.base as base
import main.ta as ta

isscaler = False

def accu(df, label, threshold):
    if threshold > 0:
        df2 = df.sort("pred", ascending = False)[:threshold]
    else:
        df2 = df.sort("pred", ascending = False)
    df2 = df2[df2["pred"] > 0.5]
    npPred = df2["pred"].values
    npLabel = df2[label].values
    npTrueInPos = npLabel[npLabel>1.0]
    return {"pos": npLabel.size, "trueInPos":npTrueInPos.size}

def get_range(df, start ,end):
    return df.query('date >="%s" & date <= "%s"' % (start, end)) 

def one_work(cls, ta, labelName, start, end, top):
    df = ta
    df = get_range(df, start, end)
    m = cls
    feat_names = base.get_feat_names(df)
    npFeat = df.loc[:,feat_names].values
    res = ""
    if isscaler :
        npFeat = s.transform(npFeat)
    topscore = None
    l = []
    for i, npPred in enumerate(m.staged_predict_proba(npFeat)):
        df.loc[:,"pred"] = npPred[:,1]
        dacc =  accu(df, labelName, top)
        acc = 0.0
        if dacc["pos"] > 0:
            acc = (dacc["trueInPos"]*1.0 / dacc["pos"])
        print i, acc
        l.append([i, acc])
    return pd.DataFrame(np.asarray(l), columns=["idx", "acc"])

def main(argv):
    clsName = argv[0]
    taName = argv[1]
    labelName = argv[2]
    start = argv[3]
    end = argv[4]
    top = int(argv[5])
    ta_father = os.path.join(root, 'data', 'ta_batch', taName )
    for d in sorted(os.listdir(ta_father)):
        if d == None or not os.path.isdir(os.path.join(ta_father,d)):
            continue
        cls = joblib.load(os.path.join(root, 'data', 'models_batch', clsName,d, "model.pkl"))
        ta = base.get_merged(os.path.join(ta_father, d))
        out_file = os.path.join(root, 'data', 'select_batch', clsName + "-" + taName + "-"  + labelName
                + "-" + start + "-" + end + "-" + str(top), d)
        if not os.path.exists(out_file):
            os.makedirs(out_file)
        res = one_work(cls, ta, labelName,start, end, top)
        res.to_csv(out_file + "/select.csv")
        print out_file

if __name__ == '__main__':
    main(sys.argv[1:])
