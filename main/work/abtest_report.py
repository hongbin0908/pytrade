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


def work(confer, f=sys.stdout, round = 0 ,topn = 10000):
    dfo = pd.read_pickle(confer.get_pred_file())
    df = dfo[(dfo.date >=confer.model_split.test_start)]
    df_sort = df.sort_values('pred', ascending=False)[["date", "sym", "open", "high", "low", "close", "pred"]]
    res = ana.accurate_topN(df, confer.scores[0], topn)
    print(res)
    #res = ana.roi_topN(df, confer.scores[0], topn)
    print("round%d : %.4f" %(round, res[res.top==topn].iloc[0, 0]), file=f)
    return res