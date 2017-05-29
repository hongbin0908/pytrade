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
from main.work import abtest_report
from main.model import ana
from main import base
from main.classifier.tree import cnn
from main.classifier.tree import ccl2
from main.classifier.tree import RFCv1n2000md3msl100
from main.classifier.ts import Ts
from main.classifier.logit2 import Logit2
from main.classifier.logit import Logit
from main.work.conf import MyConfStableLTa
from main.work.conf import MyConfForTest

from main.ta import ta_set
from keras.metrics import top_k_categorical_accuracy


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
        MyConfStableLTa(classifier=Logit2(), score=score),
    ]
def get_test_confs():
    score = 5
    return [
        MyConfForTest()
    ]

if __name__ == '__main__':
    iter_num = 2
    abtest_models = {
        "Logit10":Logit2(nb_epoch=10),
        "Logit20":Logit2(nb_epoch=20)
    }

    result_dict = {}

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-f', '--force', action='store_true',default = False, dest='force', help = 'do not use any tmp file')
    (options, args)  = parser.parse_args()

    for confer in get_confs() if not base.is_test_flag() else get_test_confs():

        for model_name in abtest_models.keys():
            result_dict[model_name] = {}
            result_dict[model_name]["exp_x2"] = 0
            result_dict[model_name]["sum_x"] = 0
            result_dict[model_name]['sum_base'] = 0
            run_model = abtest_models[model_name]
            confer.classifier = run_model
            for i in range(0, iter_num):
                confer.force = options.force
                confer.model_postfix = "abtest" + str(i)
                build.work(confer)
                score_build.work(confer)
                bitlize.work(confer)
                selected.work(confer)
                report_file = confer.get_long_report_file()
                try:
                    fd = open(report_file, "w")
                except Exception as e:
                    print("open report file error")
                    sys.exit(1)
                print(model_name, file=fd)


                model.work(confer)
                pd.set_option('display.expand_frame_repr', False)
                pd.options.display.max_rows = 999
                topn_value = 10 if base.is_test_flag() else 10000
                res = abtest_report.work(confer,f=fd, round = i, topn=topn_value)
                result_dict[model_name]['sum_x'] += res["roi3"][0]
                result_dict[model_name]['exp_x2'] += res["roi3"][0] * res["roi3"][0]
                result_dict[model_name]['sum_base'] += res['roi3'][1]
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
