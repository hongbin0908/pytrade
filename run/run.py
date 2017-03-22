#! /usr/bin/env python3.4
# -*- coding: utf-8 -*-
# @author  Bin Hong

"""
"""

import sys
import os
import platform
import socket
import numpy as np
import pandas as pd
import pickle

import matplotlib
matplotlib.use('Agg')

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

from main.work import model
from main.work import build
from main.work import score as score_build
from main.work import bitlize
from main.work import selected
from main.work import report
#from main.work import pred
from main import base
from main.work.conf import MltradeConf
from main.ta import ta_set
from main.model.spliter import YearSpliter
from main.model import ana
from main.model import feat_select
from main.classifier.tree import MyRandomForestClassifier
from main.classifier.tree import MyLogisticRegressClassifier
from main.classifier.tree import RFCv1n2000md6msl100
from main.classifier.tree import RFCv1n2000md3msl100
from main.classifier.tree import RFCv1n2000md2msl100
from main.classifier.tree import RFCv1n200md2msl100
from main.classifier.tree import RFCv1n2000md6msl10000
from main.classifier.tree import MyGradientBoostingClassifier
from main.classifier.tree import MyBayesClassifier
from main.classifier.tree import MySGDClassifier
from main.work.conf import MyConfStableLTa
from main.work.conf import MyConfForTest


if __name__ == '__main__':


    for score in [5,] :
        #confer = MyConfStableLTa(classifier=MySGDClassifier(),score=score)
        #confer = MyConfStableLTa(ta = ta_set.TaSetSma2(),     classifier=MySGDClassifier(),score=score)
        #confer = MyConfStableLTa(classifier=MyGradientBoostingClassifier(),score=score)
        #confer = MyConfStableLTa(classifier = MyLogisticRegressClassifier(max_iter=10), score=score)
        if base.is_test_flag():
            confer = MyConfForTest()
            confer.force = False
        else:
            confer = MyConfStableLTa(classifier=RFCv1n2000md6msl100(),score=score)
        confer.force = False
        build.work(confer)
        score_build.work(confer)
        bitlize.work(confer)
        selected.work(confer)
        model.work(confer)
        report.work(confer)
        dfo = pd.read_pickle(confer.get_pred_file())
        df = dfo[(dfo.date >=confer.model_split.test_start)]
        df_sort = df.sort_values('pred', ascending=False)[["date", "sym", "open", "high", "low", "close", "pred"]]
        print(df_sort[df_sort.date == base.get_last_trade_date()].head())
