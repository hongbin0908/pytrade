#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

import sys
import os
import numpy as np
import pandas as pd
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import pandas.util.testing as pdt

import matplotlib.pyplot as plt

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

import main.base as base
from main.work import conf


def cvsplit(np_data, train_num, test_num):
    i = 0
    while True:
        if i +train_num + test_num > len(np_data):
            break
        yield(np_data[i:i+train_num], np_data[i+train_num:i+train_num+test_num])
        i += test_num


def date_split(df, train_num, test_num):
    assert isinstance(df, pd.DataFrame)
    df['yyyy'] = df.date.str.slice(0,4)
    npDates = df["yyyy"].unique()
    df.set_index(["yyyy"], drop=True, inplace=True)
    for (train,test) in cvsplit(npDates, train_num, test_num):
        df_train = df.loc[train]
        df_train.reset_index(drop=False, inplace=True)
        df_test = df.loc[test]
        df_test.reset_index(drop=False, inplace=True)
        yield (df_train, df_test)

def extract_feat_label(df, scorename):
    df = df.replace([np.inf,-np.inf],np.nan).dropna()
    feat_names = base.get_feat_names(df)
    npFeat = df.loc[:,feat_names].values.copy()
    npLabel = df.loc[:,scorename].values.copy()
    return npFeat, npLabel


def split(years, split_):
    if split_[conf.SPLIT_POS_METHOD] == conf.SPLIT_METHOD_STA:
        static_test_end = len(years)-1
        while True:
            if years[static_test_end] > split_[conf.SPLIT_POS_START]:
                static_test_end -= 1
            else:
                break
        static_test_start = static_test_end  - split_[conf.SPLIT_POS_TRAIN_LEN]
    i = len(years) - 1
    while True:
        if years[i] > split_[conf.SPLIT_POS_END]:
            i -= 1
            continue
        if i+1 -split_[conf.SPLIT_POS_TEST_LEN] - split_[conf.SPLIT_POS_TRAIN_LEN] < 0:
            break
        if years[i+1-split_[conf.SPLIT_POS_TEST_LEN]] < split_[conf.SPLIT_POS_START]:
            break
        if split_[conf.SPLIT_POS_METHOD] == conf.SPLIT_METHOD_MOV:
            yield (years[i-split_[conf.SPLIT_POS_TEST_LEN]-split_[conf.SPLIT_POS_TRAIN_LEN] + 1: i-split_[conf.SPLIT_POS_TEST_LEN]+1],
                   years[i-split_[conf.SPLIT_POS_TEST_LEN] + 1:i+1])
        else:
            yield(years[static_test_start:static_test_end],
                  years[i-split_[conf.SPLIT_POS_TEST_LEN] + 1:i+1])
        i -= split_[conf.SPLIT_POS_TEST_LEN]


def post_valid(classifier, df_train, df_test, score, is_fit):
    df_train = df_train.sort_values(["sym", "date"])
    # from sklearn.exceptions import NotFittedError
    npTrainFeat, npTrainLabel = extract_feat_label(df_train, score.get_name())
    npTestFeat, npTestLabel = extract_feat_label(df_test, score.get_name())
    feat_names = base.get_feat_names(df_test)
    if not is_fit:
        probas_ = classifier.predict_proba(npTestFeat)
    else:
        classifier.fit(npTrainFeat, npTrainLabel)
        probas_ = classifier.predict_proba(npTestFeat)
    d_feat_ipts = classifier.get_feature_importances(feat_names)
    ipts = []
    for each in sorted(d_feat_ipts.items(), key=lambda a: a[1], reverse=True):
        ipts.append({"name":each[0], "score": each[1]})

    fpr, tpr, thresholds = roc_curve(npTestLabel, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    min = str(df_test.head(1)["yyyy"].values[0])
    max = str(df_test.tail(1)["yyyy"].values[0])
    df_test.loc[:, "pred"] = probas_[:, 1]
    df_test.loc[:, "pred2"] = probas_[:, 0]
    #pdt.assert_numpy_array_equal(df_test.round(2).loc[:, "pred"].values[0:10], 1 - df_test.round(2).loc[:, "pred2"].values[0:10])
    post = {"classifier": classifier,
            'ipts':ipts,
            "fpr":fpr, "tpr":tpr, 
            "thresholds": thresholds,
            "roc_auc":roc_auc,
            "name":"%s-%s" % (min, max),
            "min":min,
            "max":max,
            "df_test":df_test}
    return post


def plot_cross(cross, is_label = False):
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    for each,color in zip(cross, colors):
        if is_label:
            plt.plot(each.fpr, each.tpr, lw=2, color=color,
                    label= each.name)
        else:
            plt.plot(each.fpr, each.tpr, lw=2, color=color)
def plot_mean(mean):
    plt.plot(mean["fpr"], mean["tpr"], color='g', linestyle="--",
             label="Mean Roc (area = %0.2f)" % mean["roc_auc"], lw = 2)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k')
def plot_post(post):
    plt.plot(post.fpr, post.tpr, color='k', linestyle="-",
             label="Post Roc (area = %0.2f)" % post["roc_auc"], lw = 2)

def plot_save(outfile):
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic mltrade')
    plt.legend(loc="lower right")
    plt.savefig(outfile)
    plt.cla()
