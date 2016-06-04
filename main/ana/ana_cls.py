#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author hongbin
import sys,os
import json
import pandas as pd
from sklearn import metrics
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
from model_conf import d_conf
import model_base as base

    
def accu(npTestLabel, npPred, threshold):
    print "%f\t" % threshold,
    npPos = npPred[npPred >= threshold]
    npTrueInPos = npTestLabel[(npPred >= threshold) & (npTestLabel>1.0)]
    npTrue = npTestLabel[npTestLabel > 1.0]
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
        ddesc = json.loads(fddesc.readline())
    return ddesc

def ana(df, level, start, end):
    print "--------%s,%s-----" % (start, end)
    df = df[start : end]
    accu(df["label%d"%level].values, df["pred"].values, 0.0)
    accu(df["label%d"%level].values, df["pred"].values, 0.5)
    accu(df["label%d"%level].values, df["pred"].values, 0.55)
    accu(df["label%d"%level].values, df["pred"].values, 0.6)
    accu(df["label%d"%level].values, df["pred"].values, 0.65)
    accu(df["label%d"%level].values, df["pred"].values, 0.70)
    accu(df["label%d"%level].values, df["pred"].values, 0.75)
    accu(df["label%d"%level].values, df["pred"].values, 0.80)
        
def main(argv):
    str_conf = argv[1]
    l_conf = d_conf[str_conf]
    for each in l_conf:
        path =  os.path.join(root, 'data', 'pred', each) 
        if not os.path.isfile(os.path.join(path, "pred.csv")):
            print path, "NOT exsits"
            continue
        d_desc = read_desc(path)
        level = d_desc["level"]
        df = pd.read_csv(os.path.join(path, "pred.csv"), index_col = ['date','sym'])
        df = df.sort_index()
        print "===============%s==========" % str(d_desc)
        ana(df, level, '2010-01-01', '2015-12-31')
        ana(df, level, '2010-01-01', '2010-12-31')
        ana(df, level, '2011-01-01', '2011-12-31')
        ana(df, level, '2012-01-01', '2012-12-31')
        ana(df, level, '2013-01-01', '2013-12-31')
        ana(df, level, '2014-01-01', '2014-12-31')
        ana(df, level, '2015-01-01', '2015-12-31')
if __name__ == '__main__':
    main(sys.argv)


