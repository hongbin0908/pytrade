#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong

"""
"""

import sys
import os
import platform
import socket

import matplotlib
matplotlib.use('Agg')

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

from main.base.score2 import ScoreLabel

from main.work import model
from main.work import build
from main.work import pred
from main import base
from main.work.conf import MltradeConf
from main.ta import ta_set
from main.model.spliter import StaticSpliter
from main.classifier.tree import MyRandomForestClassifier
from main.classifier.tree import RFCv1n2000md6msl100
from main.classifier.tree import RFCv1n2000md6msl10000
from main.classifier.tree import MyGradientBoostingClassifier
from main.classifier.tree import MyLogisticRegressClassifier
from main.backtest import backtest


def getConf(index, model_split, valid_split):
    #classifier = MyGradientBoostingClassifier(n_estimators = 100)
    #classifier = RFCv1n2000md6msl100()
    classifier = MyLogisticRegressClassifier(C=1e3)
    ta = ta_set.TaSetBase1Ext4El()
    index = index

    if base.is_test_flag():
        classifier = MyRandomForestClassifier(n_estimators=10, min_samples_leaf=10)
        index = "test"

    confer = MltradeConf(150,classifier=classifier, score1=ScoreLabel(1, 1.0),
                         score2 = ScoreLabel(1, 1.0),
                         model_split=model_split,
                         valid_split=valid_split,
                         ta = ta, n_pool=30, index=index)

    return confer

if __name__ == '__main__':
    last_date = base.last_trade_date()
    confer1 = getConf("sp100_snapshot_20091129", StaticSpliter(2010,2017,1, 1700, 2010), StaticSpliter(2013, 2017, 1, 1700, 2010))
    build.work(confer1)
    model.work(confer1)


    confer2 = getConf("sp100_snapshot_20140321", StaticSpliter(2015,2017,1, 1700, 2015), StaticSpliter(2015, 2017, 1, 1700, 2015))
    build.work(confer2)
    model.work(confer2)

    


    #pred.work(confer, last_date)
    # backtest.run(os.path.join(root, "data", "cross", "pred%s.pkl" % base.last_trade_date()))

