#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

"""

"""

import sys,os
import pandas as pd
from sklearn.externals import joblib # to dump model
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)
sys.path.append(local_path)

import main.base as base

def work(classifer, df_pred):
    feat_names = base.get_feat_names(df_pred)
    np_feat = df_pred.loc[:, feat_names].values
    np_pred = classifer.predict_proba(np_feat)

    df_pred["pred"] = np_pred[:,1]
    df_pred.sort_values(['pred'], ascending=False, inplace=True)
    return df_pred[["date", "sym", "open", "high", "low", "close", "pred"]].head(20)
