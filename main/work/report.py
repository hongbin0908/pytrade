#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong

import sys
import os
import pandas as pd


local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', "..")
sys.path.append(root)

from main.model import ana

def work(confer):
    dfo = pd.read_pickle(confer.get_pred_file())
    df = dfo[(dfo.date >=confer.model_split.test_start)]
    df_sort = df.sort_values('pred', ascending=False)[["date", "sym", "open", "high", "low", "close", "pred"]]
    print(ana.roc_auc(df, confer))
    #print(ana.roc_auc_per_year(df, confer))
    print(ana.count_level(df))
    print(ana.accurate_level(df, confer.scores[0]))
    print(ana.roi_level(df, confer.scores[1]))
    print(ana.roi_level(df, confer.scores[2]))
    print(ana.roi_level_per_year(df, confer.scores[1]))
    print(ana.roi_level_per_year(df, confer.scores[2]))

