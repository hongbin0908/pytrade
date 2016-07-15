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

import main.base as base
import main.ta as ta

isscaler = False

def get_df(taName):
    dfTa = base.get_merged(os.path.join(root, 'data', 'ta', taName))
    return dfTa

def get_cls(clsName):
    cls = joblib.load(os.path.join(root, 'data', 'models',"model_" + clsName + ".pkl"))
    return cls
def get_scaler(clsName):
    return joblib.load(os.path.join(root, 'data', 'models', 'scaler_' + clsName + ".pkl"))

def select_(dfTa, top, thresh):
    #dfTa = dfTa.loc[dfTa['pred'] >= thresh]
    dfTa = dfTa.sort_values(["pred"], ascending = False).head(thresh)
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

def splay(df,top,thresh):
    df["ym"] = df.date.str.slice(0,7)
    #df2 = df.loc[df['pred'] >= thresh]
    df2 = df.sort_values(["pred"],ascending=False).head(thresh)
    df2 = df2.sort_values(["date", "pred"],ascending = False)
    df2 = df2.groupby('date').head(top)
    df2 = df2.reset_index(drop=True)
    df2["ym"] = df2.date.str.slice(0,7)
    df = df[["ym", "pred"]]
    df2 = df2[["ym", "pred"]]
    df2 = df2.groupby("ym").count()
    df = df.groupby("ym").count()
    df =  df.join(df2, lsuffix="_df1")
    for i, row in df.iterrows():
        if row["pred"] > 1:  
            pass
        else:
            print i, row["pred"]

def get_range(df, start ,end):
    return df.query('date >="%s" & date <= "%s"' % (start, end))

def main(argv):
    modelName = argv[0]
    stage = int(argv[1])
    taName = argv[2]
    batch = int(argv[3])
    start = argv[4]
    end = argv[5]
    top = argv[6]
    thresh = int(argv[7])

    ta_father = os.path.join(root, 'data', 'ta_batch', taName + "-" + str(batch))
    dfAll = None
    num  = 0
    for d in sorted(os.listdir(ta_father)):
        if d == None or not os.path.isdir(os.path.join(ta_father, d)):
            continue
        print d
        num += 1
        dfTa = base.get_merged(os.path.join(ta_father,d))
        if dfTa is None:
            continue
        dfTa = get_range(dfTa, start, end)
        if not os.path.isfile(os.path.join(root, 'data', 'models_batch',modelName,d,"model.pkl")):
            continue
        cls = joblib.load(os.path.join(root, 'data', 'models_batch',modelName,d,"model.pkl"))
        feat_names = base.get_feat_names(dfTa)
        npFeat = dfTa.loc[:,feat_names].values
        if isscaler :
            scaler = get_scaler(clsName)
            npFeatScaler = scaler.transform(npFeat)
        else:
            npFeatScaler = npFeat
        for i, npPred in enumerate(cls.staged_predict_proba(npFeatScaler)):
            if i == stage:
                break
        #npPred = cls.predict_proba(npFeat)
        dfTa["pred"] = npPred[:,1]
        dfTa = dfTa.sort_values(['pred'], ascending = False)
        print dfTa[["date","sym", "pred"]].head(1)
        print "%.2f" % (len(dfTa[dfTa["label5"] > 1.0])*1.0/len(dfTa)) ,
        dfTa = select_(dfTa, int(top), thresh)
        accu(dfTa, "label5")
        if dfAll is None:
            dfAll = dfTa
        else:
            dfAll = dfAll.append(dfTa)
        #splay(dfTa,int(top), thresh)
    dfAll = dfAll.sort_values(['pred'], ascending = False)
    print dfAll[["date", "sym", "pred"]].head()
    print "%.2f" % (len(dfAll[dfAll["label5"] > 1.0])*1.0/len(dfAll)),
    print num*thresh
    accu(select_(dfAll, int(top), num*thresh), "label5")
    splay(dfAll,int(top), batch*thresh)
if __name__ == '__main__':
    main(sys.argv[1:])
