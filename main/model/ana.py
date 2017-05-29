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


def accurate(df, score, type='long'):
    if len(df) > 0:
        if type == 'long':
            return len(df[df[score.get_name()] > 0.5]) * 1.0 /len(df)
        elif type == 'short':
            return len(df[df[score.get_name()] < 0.5]) * 1.0 /len(df)
        else:
            print('in accurate function, the type argument is wrong.expected value \
            is either long or shor, but real value is %s' %(type))
            return 0.0
    else:
        return 0.0


def count_level(df, score):
    df = df.sort_values(["pred"], ascending=False)
    columns = ["top", "date", "count", "roi"]
    res = pd.DataFrame(data=None, columns=columns)

    def cal(x):
        return pd.DataFrame({"roi":[roi2(x, score),], "count":[len(x),]})
    for top in [1000, ]:
        df_tmp = df.head(top).groupby("date").apply(lambda x: cal(x)).reset_index(drop=False)
        df_tmp['top'] = top
        df_tmp["cumcount"] = df_tmp["count"].cumsum()
        df_tmp["cumroi"] = df_tmp["roi"].cumsum()
        res = pd.concat([res, df_tmp[["top", 'date', 'count', 'roi', "cumcount", "cumroi"]]])
    res = res[["top", "date", "count", "roi", "cumcount", "cumroi"]]
    res["roip"] = res["roi"]/res["count"]
    return res
def accurate_level(df, score, type = 'long', levels = [1000,2000,3000, 5000, 10000, 100000]):
    if type == 'long':
        df = df.sort_values(["pred"], ascending=False)
    elif type == 'short':
        df = df.sort_values(["pred"], ascending=True)
    else:
        print('in accurate_level function, type argument error. expected value is \
                          either long or short, but real value is %s' % (type))
        return None
    index = ["top", "accurate", "threshold"]
    res = pd.DataFrame(data=None, columns=index)
    for top in levels:
            res = res.append(pd.Series(("%s"%top, accurate(df.head(top),score, type), \
                                        float(df.head(top).tail(1)["pred"].values)),index=index), ignore_index=True)
    return res

def accurate_topN(df, score, topN, type = 'long'):
    if type == 'long':
        df = df.sort_values(["pred"], ascending=False)
    elif type == 'short':
        df = df.sort_values(["pred"], ascending=True)
    else:
        print('in accurate_topN function, type argument error. expected value is \
                                  either long or short, but real value is %s' % (type))
        return None
    index = ["top", "accurate", "threshold"]
    res = pd.DataFrame(data=None, columns=index)
    for s in [topN, 100000]:
        if type == 'long':
            res = res.append(pd.Series(("%s"%s, accurate(df.head(s),score, type), \
                                    float(df.head(s).tail(1)["pred"].values)),index=index), ignore_index=True)
        elif type == 'short':
            res = res.append(pd.Series(("%s" % s, accurate(df.head(s), score, type), \
                                        float(df.head(s).tail(1)["pred"].values)), index=index), ignore_index=True)
        else:
            print('in accurate_topN function, type argument error. expected value is '
                  'either long or short, but real value is %s' % (type))
    return res


def get_sharp(value_x, value_base, value_sigma):
    return (value_x -value_base) * 1.0 / np.sqrt(value_sigma + 0.000001)


def roi2(df, score, max_hold_num=-1, type = 'long'):
    #在以前的操作中 已经完成了对于多空的数据按照pred的大小进行排序的逻辑, 因此, 这里默认为数据是有序的, 只需要按照多空统计roi就可以
    if max_hold_num > 0:
        df = df.groupby('date').head(max_hold_num)
    num = 1000/df.loc[:, "close"]
    if type == 'long':
        nValue = df.loc[:, "close"]*df.loc[:,score.get_name()]*num
    elif type == 'short':
        nValue = df.loc[:, "close"]*(1 - df.loc[:,score.get_name()])*num
    else:
        print('in roi2 function, type argument error. expected is either '
              'long or short, but real value is %s' %(type))
        return None
    profile = nValue - 1000
    res = float(profile.sum())
    return res


def roi(df, score, max_hold_num=-1, threshold=0, type = 'long'):
    if max_hold_num > 0:
        df = df.groupby('date').filter(lambda x: len(x)>=threshold).groupby('date').head(max_hold_num)
    num = 1000/df.loc[:, "close"]
    if type == 'long':
        nValue = df.loc[:, "close"]*df.loc[:,score.get_name()]*num
    elif type == 'short':
        nValue = df.loc[:, "close"] * ( df.loc[:, score.get_name()]) * num
    else:
        print('in roi2 function, type argument error. expected is either '
              'long or short, but real value is %s' % (type))
    if type == 'long':
        profile = nValue - 1000
    else:
        profile = 1000 - nValue

    res = float(profile.sum())
    if len(df) == 0:
        return 0, 0
    return res/len(df), len(df)

def roi_level(df, score, type = 'long', levels = [1000, 2000, 3000, 5000, -1]):
    if type == 'long':
        df = df.sort_values(["pred"], ascending=False)
    elif type == 'short':
        df = df.sort_values(['pred'], ascending= True)
    else:
        print('in roi_level function, type argument error. expected is either '
              'long or short, but real value is %s' % (type))
        return None
    index = ["top", "threshold", "num1", "roi1", "num2", "roi2", "num3", "roi3", "num4", "roi4"]
    res = pd.DataFrame(data=None, columns=index)
    for top in levels:
        if top < 0:
            df_cur = df.copy()
        else:
            df_cur = df.head(top)
        r1, num1 = roi(df_cur, score, 1, type= type)
        r2, num2 = roi(df_cur, score, 5, type= type)
        r3, num3 = roi(df_cur, score, 10, type= type)
        r4, num4 = roi(df_cur, score, -1, type= type)
        res = res.append(pd.Series((top, float(df_cur.tail(1)["pred"].values), num1, r1, num2, r2, num3, r3, num4, r4), index=index), ignore_index=True)
    return res

def roi_topN(df, score, topN, type = 'long'):
    if type == 'long':
        df = df.sort_values(["pred"], ascending=False)
    elif type == 'short':
        df = df.sort_values(["pred"], ascending=True)
    else:
        print('in roi_topN function, type argument error. expected is either '
              'long or short, but real value is %s' % (type))
        return None
    index = ["top", "threshold", "num1", "roi1", "num2", "roi2", "num3", "roi3", "num4", "roi4"]
    res = pd.DataFrame(data=None, columns=index)
    for top in [topN, -1]:
        if top < 0:
            df_cur = df.copy()
        else:
            df_cur = df.head(top)
        r1, num1 = roi(df_cur, score, 1, type = type)
        r2, num2 = roi(df_cur, score, 5, type = type)
        r3, num3 = roi(df_cur, score, 10, type = type)
        r4, num4 = roi(df_cur, score, -1, type = type)
        res = res.append(pd.Series((top, float(df_cur.tail(1)["pred"].values), num1, r1, num2, r2, num3, r3, num4, r4), index=index), ignore_index=True)
    return res

def roi_level_per_year(df, score, threshold1, threshold2, type = 'long'):
    if type == 'long':
        df = df.sort_values(["pred"], ascending=False)
    elif type == 'short':
        df = df.sort_values(['pred'], ascending= True)
    else:
        print('in roi_level_per_year function, type argument error. expected is either '
              'long or short, but real value is %s' % (type))
        return None
    index = ["year", "threshold1", "num1", 'roi1', 'threshold2',"num1", "roi1"]
    df['yyyy'] = df.date.str.slice(0,4)
    years = df["yyyy"].unique()
    res = pd.DataFrame(data=None, columns=index)
    for year in years:
        if type == 'long':
            df_cur1 = df[(df.yyyy == year) & (df.pred >= threshold1)]
        else:
            df_cur1 = df[(df.yyyy == year) & (df.pred <= threshold1)]
        roi1 = roi(df_cur1, score, type = type)[0]
        if type == 'long':
            df_cur2 = df[(df.yyyy == year) & (df.pred >= threshold2)]
        else:
            df_cur2 = df[(df.yyyy == year) & (df.pred <= threshold2)]
        roi2 = roi(df_cur2, score, type=type)[0]
        res = res.append(pd.Series((year, threshold1, len(df_cur1), roi1, threshold2, len(df_cur2), roi2), index=index), ignore_index=True)
    res = res.sort_values("year", ascending=True)
    return res

def roi_last_months(df, score, threshold1, threshold2, type = 'long'):
    if type == 'long':
        df = df.sort_values(["pred"], ascending=False)
    elif type == 'short':
        df = df.sort_values(["pred"], ascending=True)
    else:
        print('in roi_level_per_year function, type argument error. expected is either '
              'long or short, but real value is %s' % (type))
        return None
    index = ["month", "threshold1", "num1", 'roi1', 'threshold2',"num4", "roi4"]
    df['yyyyMM'] = df.date.str.slice(0,7)
    months = df["yyyyMM"].unique()
    res = pd.DataFrame(data=None, columns=index)
    for month in months:
        if type == 'long':
            df_cur1 = df[(df.yyyyMM == month) & (df.pred >= threshold1)]
        else:
            df_cur1 = df[(df.yyyyMM == month) & (df.pred <= threshold1)]
        roi1 = roi(df_cur1, score, type= type)[0]
        if type == 'long':
            df_cur2 = df[(df.yyyyMM == month) & (df.pred >= threshold2)]
        else:
            df_cur2 = df[(df.yyyyMM == month) & (df.pred <= threshold2)]
        roi2 = roi(df_cur2, score, type= type)[0]
        res = res.append(pd.Series((month, threshold1, len(df_cur1), roi1, threshold2, len(df_cur2), roi2), index=index), ignore_index=True)
    res = res.sort_values("month", ascending=True)
    return res.tail(12)

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


def _get_selected(dfTa, top, threshold, score_name, type = 'long'):
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
    sort_flag = False
    if type == 'short':
        sort_flag = True
    df2 = df.sort_values([score_name], ascending=sort_flag).groupby('date').head(top)
    if threshold > 1:
        return df2.sort_values([score_name], ascending=sort_flag).head(threshold)
    else:
        if sort_flag == False:
            return df2[df2.pred >= threshold]
        else:
            return df[df2.pred <= threshold]



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
