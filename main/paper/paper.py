#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@author  Bin Hong

import sys,os
import json
import numpy as np
import pandas as pd
from sklearn.externals import joblib # to dump model

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)
sys.path.append(local_path)

import model.modeling as  model


def get_df(taName):
    filePath = os.path.join(root, 'data', taName, "merged.pkl")
    dfTa = pd.read_pickle(filePath)
    #print "index: ", dfTa.index
    #print "columns: ", dfTa.columns
    return dfTa

def get_cls(clsName):
    cls = joblib.load(os.path.join(root, 'data', 'models',"model_" + clsName + ".pkl"))
    #print cls
    return cls
def select_(dfTa, top, start, end):
    dfTa = dfTa.loc[dfTa['date'] >= start]
    dfTa = dfTa.loc[dfTa['date'] <= end]
    #print dfTa.head()
    dfTa = dfTa.sort_values(["date", "pred"],ascending = False)
    dfTa = dfTa.groupby('date').head(top)
    return dfTa

def pre_rank(df):
    df['rank'] = np.arange(len(df)) + 1
    return df

def accu(df, label):
    npLabel = df[label].values
    npTrue = npLabel[(npLabel>1.0)]
    print npTrue.size*1.0 / npLabel.size


def main(argv):
    clsName = argv[1]
    taName = argv[2]
    start = argv[3]
    end = argv[4]
    top = argv[5]
    dfTa = get_df(taName)
    cls = get_cls(clsName)
    feat_names = model.get_feat_names(dfTa)
    npFeat = dfTa.loc[:,feat_names].values
    npPred = cls.predict_proba(npFeat)[:,1]
    dfTa["pred"] = npPred

    accu(select_(dfTa, int(top), start, end), "label5")






if __name__ == '__main__':
    main(sys.argv)