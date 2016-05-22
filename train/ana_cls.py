#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author hongbin
import sys,os
import json
import pandas as pd
from sklearn import metrics
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
import model_base as base

    
def accu(npTestLabel, npPred, threshold):
    print "%f\t" % threshold,
    npPos = npPred[npPred >= threshold]
    npTrueInPos = npTestLabel[(npPred >= threshold) & (npTestLabel==1.0)]
    npTrue = npTestLabel[npTestLabel == 1.0]
    print "%d\t%d\t" % (npPos.size, npTrueInPos.size),
    if npPos.size > 0:
        print npTrueInPos.size*1.0/npPos.size,
    else:
        print 0.0,
    print "%d\t%d\t%f\t" % (npTrue.size, npTestLabel.size, npTrue.size*1.0/npTestLabel.size)
def read_desc(path):
    fdesc = os.path.join(path, 'desc')
    assert os.path.isfile(fdesc)
    with open(fdesc, 'r') as fddesc:
        ddesc = fddesc.readline().split(',')
    level = int(ddesc[1])
    return {"level":level}

def ana(path):
    d_desc = read_desc(path)
    level = d_desc["level"]

    df = pd.read_csv(os.path.join(path, "pred.csv"), index_col = ['date','sym'])
    df = df.sort_index()
    accu(df["label%d"%level].values, df["pred"].values, 0.0)
    accu(df["label%d"%level].values, df["pred"].values, 0.5)
    accu(df["label%d"%level].values, df["pred"].values, 0.55)
    accu(df["label%d"%level].values, df["pred"].values, 0.6)
    accu(df["label%d"%level].values, df["pred"].values, 0.65)
    accu(df["label%d"%level].values, df["pred"].values, 0.70)
    accu(df["label%d"%level].values, df["pred"].values, 0.75)
    accu(df["label%d"%level].values, df["pred"].values, 0.80)
        
if __name__ == '__main__':
    ana(os.path.join(root, "data", "pred", "1002"))
    ana(os.path.join(root, "data", "pred", "1007"))
    ana(os.path.join(root, "data", "pred", "2001"))


