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
from main import base
from main.classifier.tree import cnn
from main.classifier.tree import ccl2
from main.classifier.ts import Ts
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
        MyConfStableLTa(classifier=Logit(), score=score),
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

    for confer in get_confs() if not base.is_test_flag() else get_test_confs():
        confer.force = options.force
        build.work(confer)
        score_build.work(confer)
        bitlize.work(confer)
        selected.work(confer)
        confer.force = True
        model.work(confer)
        pd.set_option('display.expand_frame_repr', False)
        pd.options.display.max_rows = 999
        report_file = os.path.join(os.path.join(local_path, '..',"data", 'report', base.get_last_trade_date() + ".txt"))
        with open(report_file, mode='w') as f:
            report.work(confer,f=f)
            dfo = pd.read_pickle(confer.get_pred_file())
            df = dfo[(dfo.date >=confer.model_split.test_start)]
            df_sort = df.sort_values('pred', ascending=False)[["date", "sym", "open", "high", "low", "close", "pred"]]
            print(df_sort[df_sort.date == base.get_last_trade_date()].head(), file=f)
#####
# 0.518290101231
#       top  accurate  threshold
# 0    1000  0.612000   0.588139
# 1    2000  0.593000   0.581140
# 2    3000  0.576667   0.576753
# 3    5000  0.560400   0.570946
# 4   10000  0.557300   0.562481
# 5  100000  0.525430   0.532426
#       top  threshold    num1      roi1    num2      roi2     num3      roi3      num4      roi4
# 0  1000.0   0.588139   191.0  5.999064   567.0  7.400042    639.0  8.971145    1000.0  8.235816
# 1  2000.0   0.581140   372.0  5.486860  1193.0  6.577869   1429.0  6.923446    2000.0  6.939116
# 2  3000.0   0.576753   521.0  5.418381  1777.0  5.617312   2190.0  5.896423    3000.0  5.957704
# 3  5000.0   0.570946   750.0  3.959670  2724.0  4.305695   3572.0  4.545629    5000.0  5.039350
# 4    -1.0   0.375443  1843.0  0.824336  9215.0  1.775203  18430.0  1.459100  673444.0  0.145319
#    year     top  threshold   num1      roi1    num2      roi2    num3      roi3     num4      roi4
# 0  2016  1000.0   0.562867  155.0 -1.338551   612.0  1.740372   821.0  3.264254   1000.0  3.742112
# 1  2016  2000.0   0.554544  191.0 -0.430061   853.0  0.459094  1379.0  1.221350   2000.0  2.938226
# 2  2016  3000.0   0.549501  217.0  1.640986   984.0  1.075356  1689.0  0.840700   3000.0  2.557243
# 3  2016  5000.0   0.542801  234.0  0.958317  1111.0  1.637218  2011.0  0.602424   5000.0  2.766848
# 4  2016    -1.0   0.375443  252.0  1.020151  1260.0  2.294831  2520.0  1.364143  89638.0  1.492326
# 0  2014  1000.0   0.571781  124.0  2.593174   468.0  1.768111   644.0  2.235027   1000.0  5.746743
# 1  2014  2000.0   0.563123  184.0  2.348788   751.0  1.048252  1120.0  1.078619   2000.0  3.921094
# 2  2014  3000.0   0.557733  205.0  2.272023   923.0  0.942571  1483.0  0.580115   3000.0  3.542883
# 3  2014  5000.0   0.550501  225.0  2.507385  1059.0  0.842574  1915.0  0.314621   5000.0  2.994418
# 4  2014    -1.0   0.400619  252.0  4.123773  1260.0  2.250202  2520.0  1.489114  94037.0  0.329919
# 0  2011  1000.0   0.563196  114.0  7.257507   469.0  6.755109   676.0  6.944542   1000.0  6.080763
# 1  2011  2000.0   0.554834  157.0  5.748535   675.0  6.353547  1085.0  6.039059   2000.0  5.438958
# 2  2011  3000.0   0.550084  179.0  5.261194   795.0  5.879678  1334.0  5.300306   3000.0  4.691160
# 3  2011  5000.0   0.543881  197.0  3.440129   924.0  6.083648  1656.0  5.175011   5000.0  3.896729
# 4  2011    -1.0   0.389759  252.0 -0.267343  1260.0  2.238469  2520.0  2.367371  92404.0 -2.571831
# 0  2013  1000.0   0.571033  131.0  4.070708   504.0  4.496119   699.0  4.207143   1000.0  6.262906
# 1  2013  2000.0   0.561799  178.0  4.156406   767.0  4.128118  1177.0  3.551992   2000.0  5.246966
# 2  2013  3000.0   0.556257  204.0  3.457860   911.0  3.698928  1517.0  3.314803   3000.0  5.029313
# 3  2013  5000.0   0.549010  230.0  4.710545  1072.0  4.249580  1913.0  3.183261   5000.0  4.419462
# 4  2013    -1.0   0.407060  252.0  4.143336  1260.0  4.054589  2520.0  3.482966  93102.0  3.406370
# 0  2015  1000.0   0.565103  123.0 -0.009782   455.0  0.219989   656.0  1.847964   1000.0  4.324655
# 1  2015  2000.0   0.556302  177.0 -2.776692   719.0 -2.628707  1075.0 -0.294568   2000.0  2.460035
# 2  2015  3000.0   0.551048  200.0 -4.121487   905.0 -3.154825  1449.0 -2.251602   3000.0  1.214763
# 3  2015  5000.0   0.544288  223.0 -5.287925  1064.0 -4.389858  1858.0 -3.790470   5000.0  0.349892
# 4  2015    -1.0   0.384898  252.0 -3.424377  1260.0 -3.480185  2520.0 -3.199466  92511.0 -3.643288
# 0  2012  1000.0   0.566705  142.0  1.772087   522.0  4.447893   700.0  5.304251   1000.0  7.822448
# 1  2012  2000.0   0.557989  188.0  0.745040   805.0  3.757737  1212.0  5.088005   2000.0  7.668812
# 2  2012  3000.0   0.552759  222.0 -0.830723   985.0  2.535272  1585.0  3.931781   3000.0  6.682119
# 3  2012  5000.0   0.546215  237.0 -1.239109  1132.0  1.963497  2017.0  2.883394   5000.0  6.275678
# 4  2012    -1.0   0.393594  250.0 -1.359304  1250.0  1.517594  2500.0  1.847942  90284.0  0.962672
# 0  2017  1000.0   0.550083   80.0  4.613985   376.0  3.735758   636.0  1.667364   1000.0  2.031291
# 1  2017  2000.0   0.541506   81.0  4.150799   403.0  2.990488   778.0  1.407785   2000.0  1.534718
# 2  2017  3000.0   0.536087   81.0  4.150799   405.0  3.074233   802.0  1.436367   3000.0  1.795817
# 3  2017  5000.0   0.529514   81.0  4.150799   405.0  3.074233   810.0  1.452037   5000.0  1.855943
# 4  2017    -1.0   0.403678   81.0  4.150799   405.0  3.074233   810.0  1.452037  28670.0  0.847391
# 0  2010  1000.0   0.561536  141.0 -0.773553   531.0  3.559727   711.0  4.111708   1000.0  1.090504
# 1  2010  2000.0   0.554119  192.0 -0.936323   810.0  3.459390  1247.0  4.518793   2000.0  2.656532
# 2  2010  3000.0   0.549509  215.0  0.456514   982.0  3.627568  1615.0  3.681119   3000.0  1.846515
# 3  2010  5000.0   0.543586  228.0  0.210612  1084.0  3.927246  1976.0  3.956048   5000.0  2.464812
# 4  2010    -1.0   0.385454  252.0  0.447567  1260.0  3.131334  2520.0  2.866985  92798.0  0.855769

