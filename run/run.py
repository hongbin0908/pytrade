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
#from main.work import pred
from main import base
from main.work.conf import MltradeConf
from main.ta import ta_set
from main.model.spliter import BinarySpliter
from main.classifier.tree import MyRandomForestClassifier
from main.classifier.tree import RFCv1n2000md6msl100
from main.classifier.tree import RFCv1n2000md6msl10000
from main.classifier.tree import MyGradientBoostingClassifier
from main.classifier.tree import MyLogisticRegressClassifier
from main.backtest import backtest


def getConf(week,
        index="sp100_snapshot_20091129",
        model_split=BinarySpliter("2010-01-01", "2017-01-01", "1700-01-01", "2010-01-01")
):
    #classifier = MyGradientBoostingClassifier(n_estimators = 100)
    #classifier = RFCv1n2000md6msl100()
    index = index
    classifier = MyLogisticRegressClassifier(C=1e3)
    ta = ta_set.TaSetBase1Ext4El()
    index = index
    week = week
    if base.is_test_flag():
        classifier = MyRandomForestClassifier(n_estimators=10, min_samples_leaf=10)
        index = "test"

    confer = MltradeConf(
                         model_split=model_split,
                         classifier=classifier,
                         score1=ScoreLabel(5, 1.0),
                         score2 = ScoreLabel(5, 1.0),
                         ta = ta, n_pool=30, index=index, week = week)

    return confer

if __name__ == '__main__':
    last_date = base.last_trade_date()
    confer1 = getConf(1, index = "sp100_snapshot_20091129", model_split=BinarySpliter("2010-01-01", "2013-01-01", "2000-01-01", "2010-01-01"))
    build.work(confer1)
    model.work(confer1)

    confer2 = getConf(1, index = "sp100_snapshot_20120316", model_split=BinarySpliter("2013-01-01", "2015-01-01", "2003-01-01", "2013-01-01"))
    build.work(confer2)
    model.work(confer2)

    confer3 = getConf(1, index = "sp100_snapshot_20140321", model_split=BinarySpliter("2015-01-01", "2017-01-01", "2005-01-01", "2015-01-01"))
    build.work(confer3)
    model.work(confer3)

    # backtest.run(os.path.join(root, "data", "cross", "pred%s.pkl" % base.last_trade_date()))

