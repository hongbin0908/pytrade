#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

#@author Bin Hong
import os,sys
import pandas as pd
import multiprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib # to dump model
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

import main.base as base
import main.ta as ta
import main.yeod.yeod as yeod
import main.ta.build as build

def ana(df, df2):
    df.loc[:,'yyyy'] = df.date.str.slice(0,4)
    df2.loc[:,'yyyy'] = df2.date.str.slice(0,4)
    dfTrue = df[df["label5"] > 1.0]
    df2True = df2[df2["label5"] > 1.0]
    re =  df.groupby('yyyy').count()\
          .join(dfTrue.groupby('yyyy').count(), rsuffix='_dfTrue')\
          .join(df2.groupby('yyyy').count(),rsuffix='_df2')\
          .join(df2True.groupby('yyyy').count(), rsuffix='_df2True')
    re = re[['label5','label5_dfTrue', 'label5_df2','label5_df2True']]
    re.loc[:,"rate1"] = re["label5_dfTrue"]*1.0/re["label5"]
    re.loc[:,"rate2"] = re["label5_df2True"]*1.0/re["label5_df2"]
    return re

def main(args):
    exec "import main.pandas_talib.sig_%s as conf" % args.signame
    build.work2(20, 'sp500Top50', args.signame)
    df = base.get_merged(conf.__name__, yeod.get_sp500Top50())
    df.to_csv("ta.csv")

    tree = DecisionTreeClassifier() 
    
    feat_names = base.get_feat_names(df)

    dfTrain = df[(df.date>='1970-01-01') & (df.date <='2009-12-31')]
    npTrainFeat = dfTrain.loc[:,feat_names].values.copy()
    npTrainLabel = dfTrain.loc[:,"label5"].values.copy()
    npTrainLabel[npTrainLabel >  1.0] = 1
    npTrainLabel[npTrainLabel <  1.0] = 0

    tree.fit(npTrainFeat, npTrainLabel)
    joblib.dump(tree, "tree.pkl", compress = 3)
    
    dfTest = df[(df.date>='2010-01-01') & (df.date <='2099-12-31')]
    npTestFeat = dfTest.loc[:, feat_names].values.copy()
    
    npPred = tree.predict_proba(npTestFeat)

    dfTest.loc[:,"pred"] = npPred[:,1]
    
    print dfTest['pred'].head()

    dfPos = dfTest[ dfTest['pred'] > 0.55 ]
    print 1.0 * len(dfPos[dfPos['label5']>1])  / len(dfPos)
    print 1.0 * len(dfTest[dfTest['label5']>1])  / len(dfTest)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='check test')
    parser.add_argument('signame', help = "sig file name without 'sig_'")
    args = parser.parse_args()
    main(args)

