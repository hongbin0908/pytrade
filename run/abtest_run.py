#! /usr/bin/env python3.4
# -*- coding: utf-8 -*-
# @author  Bin Hong

import sys
import os
import numpy as np
np.random.seed(608317)
import pandas as pd

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

from main.work import model
from main.work import build
from main.work import score as score_build
from main.work import bitlize
from main.work import selected
from main.work import abtest_report
from main.model import ana
from main import base
from main.classifier.logit2 import Logit2
from main.classifier.tf_dnn import TfDnn
from main.work.conf import MyConfStableLTa

if __name__ == '__main__':
    iter_num = 1
    abtest_confs = {
        #"adj": MyConfStableLTa(classifier=Logit2(30), is_adj= True),
        "score5": MyConfStableLTa(classifier=Logit2(30), is_adj = False),
        "score4": MyConfStableLTa(classifier=Logit2(30), is_adj = False, score=4),
        "score6": MyConfStableLTa(classifier=Logit2(30), is_adj = False, score=6),
        "TfDnn1":  MyConfStableLTa(classifier=TfDnn(nb_epoch=30),is_adj=False, score=5),
        "TfDnn2":  MyConfStableLTa(classifier=TfDnn(nb_epoch=20),is_adj=False, score=5),
        "TfDnn3":  MyConfStableLTa(classifier=TfDnn(nb_epoch=10),is_adj=False, score=5),
        "TfDnn4":  MyConfStableLTa(classifier=TfDnn(nb_epoch=40),is_adj=False, score=5),
        #"score2": MyConfStableLTa(classifier=Logit2(30), is_adj = False, score=2),
        #"score8": MyConfStableLTa(classifier=Logit2(30), is_adj = False, score=8),
        #"score32": MyConfStableLTa(classifier=Logit2(30), is_adj = False, score=32),
    }
    result_dict = {}

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-f', '--force', action='store_true',default = False, dest='force', help = 'do not use any tmp file')
    (options, args)  = parser.parse_args()

    report_file = os.path.join(root, 'data', 'report', 'abtest.txt')
    fd = open(report_file, "w")
    for model_name in abtest_confs.keys():
        print(model_name, file = fd)
        result_dict[model_name] = {}
        result_dict[model_name]["exp_x2"] = 0
        result_dict[model_name]["sum_x"] = 0
        result_dict[model_name]['sum_base'] = 0
        confer = abtest_confs[model_name]
        for i in range(0, iter_num):
            confer.force = options.force
            confer.model_postfix = "abtest" + str(i)
            print(confer.get_pred_file())
            build.work(confer)
            score_build.work(confer)
            bitlize.work(confer)
            selected.work(confer)
            print(confer.get_classifier_file())
            model.work(confer)
            pd.set_option('display.expand_frame_repr', False)
            pd.options.display.max_rows = 999
            topn_value = 10 if base.is_test_flag() else 1000
            res = abtest_report.work(confer,f=fd, round = i, topn=topn_value)
            result_dict[model_name]['sum_x'] += res["accurate"][0]
            result_dict[model_name]['exp_x2'] += res["accurate"][0] * res["accurate"][0]
            result_dict[model_name]['sum_base'] += res['accurate'][1]
            """
            dfo = pd.read_pickle(confer.get_pred_file())
            df = dfo[(dfo.date >=confer.model_split.test_start)]
            df_sort = df.sort_values('pred', ascending=False)[["date", "sym", "open", "high", "low", "close", "pred"]]
            #print(df_sort[df_sort.date == confer.last_trade_date].head(), file=fd)
            """
    print("summary", file = fd)
    for i in result_dict:
        value_x = result_dict[i]['sum_x'] * 1.0 / iter_num
        cov_x = (result_dict[i]['exp_x2'] - iter_num * value_x * value_x) * 1.0/iter_num
        value_base = result_dict[i]['sum_base'] * 1.0 / iter_num
        sharp_value = ana.get_sharp(value_x= value_x, value_base= value_base, value_sigma= cov_x)
        print("model %s: avg = %.4f, cov = %.4f, sharp_index = %.6f" %(i, value_x, cov_x, sharp_value), file = fd)
    fd.close()
