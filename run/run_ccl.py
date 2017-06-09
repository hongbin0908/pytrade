#! /usr/bin/env python3.4
# -*- coding: utf-8 -*-
# @author  Bin Hong

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

from main.work import model
from main.work import build
from main.work import score as score_build
from main.work import bitlize
from main.work import selected
from main.work import report
from main.work import short_report
from main import base
from main.classifier.tree import cnn
from main.classifier.tree import ccl2
from main.classifier.ts import Ts
from main.classifier.logit2 import Logit2
from main.work.conf import MyConfStableLTa
from main.work.conf import MyConfForTest
from main.work.conf import MyMdnConfForTest
from main.dassert import dassert_yeod
from main.dassert import dassert_ta


def get_confs2():
    score = 5
    return [
        MyConfStableLTa(classifier=Ts(max_iterations=20000), score=score),
    ]
def get_confs1():
    score = 5
    return [
        MyConfStableLTa(classifier=cnn(nb_epoch=5), score=score),
        MyConfStableLTa(classifier=cnn(),score=score),
        MyConfStableLTa(classifier=cnn(num_filt_2=4), score=score),
        MyConfStableLTa(classifier=cnn(num_filt_1=6, num_filt_2=4), score=score),
        MyConfStableLTa(classifier=cnn(batch_size=1000), score=score),
    ]

def get_confs2():
    score = 5
    return [
        MyConfStableLTa(classifier=ccl2(batch_size=32, nb_epoch=10), score=score),
    ]
def get_confs():
    score = 5
    return [
        #MyConfStableLTa(classifier=ccl2(batch_size=32, nb_epoch=20), score=score),
        #MyConfStableLTa(classifier=cnn(batch_size=32, nb_epoch=20), score=score),
        #MyConfStableLTa(classifier=Logit2(), score=score),
        MyConfStableLTa(classifier=Logit2(nb_epoch=30), score=score),
    ]

def get_mdnconfs():
    score = 5
    inputsiz = 150
    hidden_size = 50
    model_size = 200
    lr = 0.0015
    train_Begin = '2005'
    train_end = '2010'
    test_begin = '2010'
    test_end = '2011'
    return [
        MyMdnConfForTest(inputsiz= inputsiz, hidden_size= hidden_size,
                         model_size=model_size, lr=lr,
                         train_Begin=train_Begin, train_end=train_end,
                         test_begin=test_begin, test_end=test_end),
    ]

def get_test_confs():
    score = 5
    return [
        MyConfForTest()
    ]



if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-f', '--force', action='store_true',default = False, dest='force', help = 'do not use any tmp file')
    (options, args)  = parser.parse_args()

    for confer in get_confs() if not base.is_test_flag() else get_mdnconfs():
        confer.force = options.force
        confer.force = True
        if not base.is_test_flag():
            dassert_yeod.work(confer)
        build.work(confer)

        score_build.work_with_original_fea(confer)
        confer.force = True
        print(confer.get_classifier_file())
        model.work_with_original_Fea(confer)
        pd.set_option('display.expand_frame_repr', False)
        pd.options.display.max_rows = 999
        long_report_file = confer.get_long_report_file()
        short_report_file = confer.get_short_report_file()

        with open(long_report_file, mode='w') as f:
            report.work(confer,f=f)
            dfo = pd.read_pickle(confer.get_pred_file())
            df = dfo[(dfo.date >=confer.model_split.test_start)]
            df_sort = df.sort_values('pred', ascending=False)[["date", "sym", "open", "high", "low", "close", "pred"]]
            print(df_sort[df_sort.date == confer.last_trade_date].head(), file=f)

        with open(short_report_file, mode = 'w') as f:
            short_report.work(confer, f=f)
            dfo = pd.read_pickle(confer.get_pred_file())
            df = dfo[(dfo.date >= confer.model_split.test_start)]
            df_sort = df.sort_values('pred', ascending=True)[["date", "sym", "open", "high", "low", "close", "pred"]]
            print(df_sort[df_sort.date == confer.last_trade_date].head(), file=f)

