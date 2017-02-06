#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong

"""
"""

import sys
import os
import platform
import socket
import pickle
import pandas as pd
import numpy as np

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..',"..")
sys.path.append(root)

from main import base
from main.base.score2 import ScoreLabel
from main.work import model
from main.work import build
from main.work.conf import MyConfStableLTa
from main.model import ana

confer = MyConfStableLTa()

#build.work(confer)
#model.work(confer)



df = pd.read_pickle(os.path.join(root, 'output', "result_20170205.pkl"))
print(ana.roc_auc(df, confer))

clazz_file_name = confer.get_classifier_file()
with open(clazz_file_name, 'rb') as fin:
    clazz = pickle.load(fin)

feat_names = base.get_feat_names(df)
ipts = sorted(clazz.get_feature_importances(feat_names).items(), key=lambda a:a[1], reverse=True)
for each in ipts:
    print(each)


dfo = df.sort_values("pred", ascending=False)
df = dfo[feat_names]
df_sum = df.sum(axis=0).to_frame(name='sum')
df_sum = df_sum/len(df)
print(df_sum.sort_values("sum", ascending=False).head())

df_top = dfo[dfo[confer.score1.get_name()]==0].head(100)[feat_names]
df_top_sum = df_top.sum(axis=0).to_frame(name='sum')
df_top_sum = (df_top_sum/len(df_top)).sort_values("sum", ascending=False)
print(df_top_sum.head())
print(df_top_sum.tail())

df_top2 = dfo[dfo[confer.score1.get_name()]==1].head(100)[feat_names]
df_top2_sum = df_top2.sum(axis=0).to_frame(name='sum')
df_top2_sum = (df_top2_sum/len(df_top2)).sort_values("sum", ascending=False)
print(df_top2_sum.head())
print(df_top2_sum.tail())

df_merge = df_top_sum.join(df_top2_sum, rsuffix='2')

df_merge["diff"] = np.abs(df_merge["sum2"].values - df_merge["sum"].values)
df_merge = df_merge.sort_values("diff")
pd.options.display.max_rows = 999
print(df_merge.tail(100))

df_merge = df_merge.join(df_sum, rsuffix='3')

df_merge["diff"] = np.abs(df_merge["sum3"].values - df_merge["sum"].values)
df_merge = df_merge.sort_values("diff")
pd.options.display.max_rows = 999
print(df_merge.tail(1000))

#for i,each in df.head(100).iterrows():
#    print(each["sym"], each["pred"])

