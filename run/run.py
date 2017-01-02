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
from main.work.conf import MltradeConf
from main.ta import ta_set
from main.model.spliter import StaticSpliter
from main.classifier.tree import MyRandomForestClassifier
from main.classifier.tree import RFCv1n2000md6msl100
from main.classifier.tree import MyGradientBoostingClassifier


if platform.platform().startswith("Windows"):
    TEST = True
elif platform.platform().startswith("Darwin"):
    TEST = True
elif '47.90.41.27' == socket.gethostbyname(socket.gethostname()):
    TEST = True
else:
    TEST = False

def getConf2():
    """
    for test
    :return:
    """
    classifier = MyRandomForestClassifier()
    confer = MltradeConf(500,classifier=classifier, score1=ScoreLabel(5, 1.0),
                         score2 = ScoreLabel(5, 1.0),
                         model_split=StaticSpliter(2010,2013,1, 1900, 2010),
                         valid_split=StaticSpliter(2013, 2017, 1, 1900, 2010),
                         ta = ta_set.TaSetBase1Ext8(), n_pool=25)
    return confer

def getConf():
    if not TEST:
        #classifier = MyRandomForestClassifier(n_estimators = 1000)
        classifier = MyGradientBoostingClassifier(n_estimators = 100)
        classifier = RFCv1n2000md6msl100()
        ta = ta_set.TaSetBase1Ext4El()
        confer = MltradeConf(150,classifier=classifier, score1=ScoreLabel(5, 1.0),
                             score2 = ScoreLabel(5, 1.0),
                             model_split=StaticSpliter(2010,2017,1, 1700, 2010),
                             valid_split=StaticSpliter(2013, 2017, 1, 1700, 2010),
                             ta = ta, n_pool=30, index="sp100")
        confer.syms = confer.syms[0:1]

    else:
        ta = ta_set.TaSetBase1()
        confer = MltradeConf(2,
                classifier=MyRandomForestClassifier(n_estimators=10, min_samples_leaf=10), 
                score1=ScoreLabel(5, 1.0),
                score2 = ScoreLabel(5, 1.0),
                model_split=StaticSpliter(2010,2013, 1, 2000, 2010),
                valid_split=StaticSpliter(2013, 2017, 1, 2003, 2013),
                ta = ta, n_pool=1)
        confer.syms = confer.syms[0:1]
    return confer

if __name__ == '__main__':
    confer = getConf()
    build.work(confer)
    model.work(confer)
    pred.work(confer)
