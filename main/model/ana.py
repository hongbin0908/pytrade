#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

import sys
import os
import pandas as pd
import numpy as np

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

from main import base


def print_empty(df, fout):
    for each in df.iterrows():
        pred = each[1]["pred"]
        pred_dfTrue = each[1]["pred_dfTrue"]
        if np.isnan(pred_dfTrue):
            pre_dfTrue = 0
        pred_df2 = each[1]["pred_df2"]
        if np.isnan(pred_df2):
            pred_df2 = 0
        pred_df2True = each[1]["pred_df2True"]
        if np.isnan(pred_df2True):
            pred_df2True = 0
        if pred_df2 > 0:
            continue
        print("%s" % each[1]["yyyyMM"], file=fout)


def group_by_year(df, df2, score1, score2):
    df['yyyy'] = df.date.str.slice(0,4)
    df_1 =  _group_by_xxxx(df,df2,score1.get_name(),"yyyy")
    df_2 =  _group_by_xxxx(df,df2,score2.get_name(),"yyyy")
    return df_1.join(df_2, lsuffix="_label1", rsuffix="_label2")


def group_by_month(df, df2, score1, score2):
    df["yyyyMM"] = df.date.str.slice(0,7)
    df_1 =  _group_by_xxxx(df,df2,score1.get_name(),"yyyyMM")
    df_2 =  _group_by_xxxx(df,df2,score2.get_name(),"yyyyMM")
    return df_1.join(df_2, lsuffix="_label1", rsuffix="_label2")


def accurate(df, score):
    if len(df) > 0:
        return len(df[df[score.get_name()] == 1]) * 1.0 /len(df)
    else:
        return 0.0


def _group_by_xxxx(df, df2, scorename, xxxx):
    dfTrue = df[df[scorename] == 1]
    df2True = df2[df2[scorename] == 1]
    re = df.groupby(xxxx).count() \
        .join(dfTrue.groupby(xxxx).count(), rsuffix='_dfTrue') \
        .join(df2.groupby(xxxx).count(), rsuffix='_df2') \
        .join(df2True.groupby(xxxx).count(), rsuffix='_df2True')
    re = re[['pred', 'pred_dfTrue', 'pred_df2', 'pred_df2True']]
    re["rate1"] = re["pred_dfTrue"] * 1.0 / re["pred"]
    re["rate2"] = re["pred_df2True"] * 1.0 / re["pred_df2"]
    return re


def _get_selected(dfTa, top, threshold, score_name):
    """
    :param dfTa: dataframe(["date", "sym", "pred", "pred2"])
            where: pred2 = 1- pred
    :param top: int select top good sym everyday
    :param threshold:
           when *threshold* <1 select which *score_name* > pred
           when *threshold* >1 select which num <= threshold
    :param score_name: the value the threshold to compare ["pred", "pred2"]
    :return: the selected dataframe
    e.g.
    date        sym     pred    pred2
    20160101    GOOG    0.6     0.4
    20160101    MSFT    0.5     0.5
    20160101    YHOO    0.4     0.6
    20160102    GOOG    0.6     0.4
    20160102    MSFT    0.5     0.5
    20160102    YHOO    0.4     0.6

    when top = 2, threshold = 0.6, score_name = pred
    result:
    date        sym     pred    pred2
    20160101    GOOG    0.6     0.4
    20160102    GOOG    0.6     0.4

    why need 2 param to control selection?
    1. the threshold control the quality of the seletion
    2. the top control the 均匀度
    """

    df = dfTa
    df['yyyy'] = df.date.str.slice(0,4)
    df["yyyyMM"] = df.date.str.slice(0,7)
    df2 = df.sort_values([score_name], ascending=False).groupby('date').head(top)
    if threshold > 1:
        return df2.sort_values([score_name], ascending=False).head(threshold)
    else:
        return df2[df2.pred >= threshold]


def select2(score1, score2, df_test, top, threshold, score_name):
    df_selected = _get_selected(df_test, top, threshold, score_name)
    df_year = group_by_year(df_test, df_selected, score1, score2)
    df_year.reset_index(drop=False, inplace=True)
    df_month = group_by_month(df_test, df_selected, score1, score2)
    df_month.reset_index(drop=False, inplace=True)
    assert isinstance(df_year, pd.DataFrame)
    df_year = df_year.fillna(0)
    return df_selected, df_year, df_month
