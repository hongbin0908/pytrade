#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong

"""
"""

import sys
import os
import platform
import socket
import pandas as pd

import matplotlib
matplotlib.use('Agg')

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

from main.base.score2 import ScoreLabel
from main.work import model
from main.work import build
#from main.work import pred
from main import base
from main.work.conf import MltradeConf
from main.ta import ta_set
from main.model.spliter import YearSpliter
from main.model import ana
from main.classifier.tree import MyRandomForestClassifier
from main.classifier.tree import RFCv1n2000md6msl100
from main.classifier.tree import RFCv1n2000md6msl10000
from main.classifier.tree import MyGradientBoostingClassifier
from main.classifier.tree import MyLogisticRegressClassifier
from main.backtest import backtest


def run(classifier,  output, train_start = "1980", score=5, week = -1,
        index="sp100_snapshot_20091129"
):
    model_split=YearSpliter("2010", "2017", train_start, "2010")
    #classifier = MyGradientBoostingClassifier(n_estimators = 100)
    classifier = classifier
    index = index
    #classifier = MyLogisticRegressClassifier(C=1e3)
    ta = ta_set.TaSetBase1Ext4El()
    index = index
    week = week
    if base.is_test_flag():
        classifier = MyRandomForestClassifier(n_estimators=10, min_samples_leaf=10)
        index = "test"

    confer = MltradeConf(
                         model_split=model_split,
                         classifier=classifier,
                         score1=ScoreLabel(score, 1.0),
                         score2 = ScoreLabel(score, 1.0),
                         ta = ta, n_pool=30, index=index, week = week)

    build.work(confer)
    model.work(confer)
    df = pd.read_pickle(confer.get_pred_file())
    df = df[(df.date >=confer.model_split.test_start) & (df.date<=confer.model_split.test_end)]
    df.to_pickle(os.path.join(root, 'output', output))
    return confer

if __name__ == '__main__':
    confer = run(RFCv1n2000md6msl100(), "result_20170204_rf_s5.pkl")
    df = pd.read_pickle(os.path.join(root, 'output', "result_20170204_rf_s5.pkl"))
    print(ana.roc_auc(df, confer))
    print(ana.roc_auc_per_year(df, confer))

    confer = run(RFCv1n2000md6msl100(), "result_20170204_rf_s5_1900.pkl", train_start="1900")
    df = pd.read_pickle(os.path.join(root, 'output', "result_20170204_rf_s5_1900.pkl"))
    print(ana.roc_auc(df, confer))
    print(ana.roc_auc_per_year(df, confer))

    confer = run(MyGradientBoostingClassifier(n_estimators = 100), "result_20170204_gb_s5.pkl")
    df = pd.read_pickle(os.path.join(root, 'output', "result_20170204_gb_s5.pkl"))
    print(ana.roc_auc(df, confer))
    print(ana.roc_auc_per_year(df, confer))

    confer = run(RFCv1n2000md6msl100(), "result_20170204_rf_s1.pkl", score=1)
    df = pd.read_pickle(os.path.join(root, 'output', "result_20170204_rf_s1.pkl"))
    print(ana.roc_auc(df, confer))
    print(ana.roc_auc_per_year(df, confer))

    confer = run(MyGradientBoostingClassifier(n_estimators = 100), "result_20170204_gb_s1.pkl", score=1)
    df = pd.read_pickle(os.path.join(root, 'output', "result_20170203_gb_s1.pkl"))
    print(ana.roc_auc(df, confer))
    print(ana.roc_auc_per_year(df, confer))

