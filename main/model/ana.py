#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc,precision_recall_curve
import matplotlib.pyplot as plt

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
        return len(df[df[score.get_name()] > 0.5]) * 1.0 /len(df)
    else:
        return 0.0

def count_level(df, score):
    df = df.sort_values(["pred"], ascending=False)
    columns = ["top", "date", "count", "roi"]
    res = pd.DataFrame(data=None, columns=columns)

    def cal(x):
        return pd.DataFrame({"roi":[roi(x, score)[0],], "count":[len(x),]})
    for top in [10, ]:
        df_tmp = df.head(top).groupby("date").apply(lambda x: cal(x)).reset_index(drop=False)
        df_tmp['top'] = top
        res = pd.concat([res, df_tmp[["top", 'date', 'count', 'roi']]])
    res = res[["top", "date", "count", "roi"]]
    res["cumcount"] = res["count"].cumsum()
    res["cumroi"] = res["roi"].cumsum()
    return res
def accurate_level(df, score):
    df = df.sort_values(["pred"], ascending=False)
    index = ["top", "accurate", "threshold"]
    res = pd.DataFrame(data=None, columns=index)
    for top in [1000, 5000, 10000, 100000]:
        res = res.append(pd.Series(("%s"%top, accurate(df.head(top),score), float(df.head(top).tail(1)["pred"].values)),index=index), ignore_index=True)
    return res


def roi2(df, score, max_hold_num=-1):
    if max_hold_num > 0:
        df = df.groupby('date').head(max_hold_num)
    num = 1000/df.loc[:, "close"]
    nValue = df.loc[:, "close"]*df.loc[:,score.get_name()]*num
    profile = nValue - 1000
    res = float(profile.sum())
    return res

def roi(df, score, max_hold_num=-1):
    if max_hold_num > 0:
        df = df.groupby('date').head(max_hold_num)
    num = 1000/df.loc[:, "close"]
    nValue = df.loc[:, "close"]*df.loc[:,score.get_name()]*num
    profile = nValue - 1000
    res = float(profile.sum())
    return res/len(df), len(df)

def roi_level(df, score):
    df = df.sort_values(["pred"], ascending=False)
    index = ["top", "threshold", "num1", "roi1", "num2", "roi2", "num3", "roi3", "num4", "roi4"]
    res = pd.DataFrame(data=None, columns=index)
    for top in [10, 1000, 5000, 10000, 100000, -1]:
        if top < 0:
            df_cur = df.copy()
        else:
            df_cur = df.head(top)
        if top == 1000:
            df_tmp = df_cur.groupby('date').head(1000)

        r1, num1 = roi(df_cur, score, 1)
        r2, num2 = roi(df_cur, score, 5)
        r3, num3 = roi(df_cur, score, 10)
        r4, num4 = roi(df_cur, score, -1)
        res = res.append(pd.Series((top, float(df_cur.tail(1)["pred"].values), num1, r1, num2, r2, num3, r3, num4, r4), index=index), ignore_index=True)
    return res

def roi_level_per_year(df, score):
    df = df.sort_values(["pred"], ascending=False)
    index = ["year", "top", "threshold", "num1", "roi1", "num2", "roi2", "num3", "roi3", "num4", "roi4"]
    df['yyyy'] = df.date.str.slice(0,4)
    years = df["yyyy"].unique()
    res = pd.DataFrame(data=None, columns=index)
    for year in years:
        df_cur = df[df.yyyy == year]
        roi = roi_level(df_cur, score)
        roi["year"] = year
        res = res.append(roi)
        #if year == "2017":
        #    print(df_cur.head(100)[["open", "high", "low", "close", "pred", score.get_name()]])
        #    sys.exit(0)
    #res = res.append(pd.Series(("all", "total", accurate(df,score), 0.5),index=index), ignore_index=True)
    return res[["year", "top", "threshold", "num1", "roi1", "num2", "roi2", "num3", "roi3", "num4", "roi4"]]

def accurate_level_per_year(df, score):
    index = ["year", "top", "accurate", "threshold"]
    res = pd.DataFrame(data=None, columns=index)
    years = df["yyyy"].unique()
    for year in years:
        df_cur = df[df.yyyy == year]
        acc = accurate_level(df_cur, score)
        acc["year"] = year
        res = res.append(acc)
    return res


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

def extract_feat_label(df, scorename):
    df = df.replace([np.inf,-np.inf],np.nan).dropna()
    feat_names = base.get_feat_names(df)
    npFeat = df.loc[:,feat_names].values.copy()
    npLabel = df.loc[:,scorename].values.copy()
    npPred = df.loc[:, "pred"].values.copy()
    return npFeat, npLabel, npPred

def roc_auc(df, confer):
    npFeat, npLabel, npPred = extract_feat_label(df, confer.scores[0].get_name())
    fpr, tpr, thresholds = roc_curve(npLabel, npPred)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def roc_auc_per_year(df, confer):
    res = []
    df['yyyy'] = df.date.str.slice(0,4)
    years = df["yyyy"].unique()
    for year in years:
        df_cur = df[df.yyyy == year]
        res.append({"yyyy":year, "roc":roc_auc(df_cur,confer)})
    return pd.DataFrame(data = res).sort_values("yyyy")

def plot_roc(df, confer, outfile):
    npFeat, npLabel, npPred = extract_feat_label(df, confer.score1.get_name())
    fpr, tpr, thresholds = roc_curve(npLabel, npPred)
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0,1],[0,1], linestyle='--', lw=1, color='k')
    plt.savefig(outfile)
    plt.cla()

def plot_precision_recall_per_year(df, confer, outfile):
    df['yyyy'] = df.date.str.slice(0,4)
    years = df['yyyy'].unique()
    for year in years:
        df_cur = df[df.yyyy == year]
        npFeat, npLabel, npPred = extract_feat_label(df_cur, confer.score1.get_name())
        precision, recall, thresholds = precision_recall_curve(npLabel, npPred)
        plt.plot(recall, precision, lw=2)
    plt.savefig(outfile)
    plt.cla()
