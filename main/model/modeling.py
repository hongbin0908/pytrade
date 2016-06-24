#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

import sys,os
import json
import numpy as np
import pandas as pd
import math
import multiprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib # to dump model

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
print root
sys.path.append(root)

import main.base as base
import main.ta as ta

def build_trains(df, start, end):
    """
    param sym2feats:
        dict from symbol to its features dataframe
    param start:
        the first day(include) to merge
    param end:
        the last day(exclude) to merge
    """
    df = df[ (df.date >= start) & (df.date<=end)]
    return df

def one_work(name, dir_ta, model, label, date_range):
    if os.path.isfile(os.path.join(root, 'data', 'models', "model_%s.pkl" % name)):
        print "%s already exists!" % name
        return
    dfTa = ta.get_merged(dir_ta)
    dfTrain = build_trains(dfTa, date_range[0], date_range[1])
    feat_names = base.get_feat_names(dfTrain)
    npTrainFeat = dfTrain.loc[:,feat_names].values
    npTrainLabel = dfTrain.loc[:,label].values.copy()
    npTrainLabel[npTrainLabel != 1.0]
    npTrainLabel[npTrainLabel >  1.0] = 1
    npTrainLabel[npTrainLabel <  1.0] = 0
    model.fit(npTrainFeat, npTrainLabel)
    joblib.dump(model, os.path.join(root, "data", "models", "model_%s.pkl" % name), compress = 3)
    dFeatImps = dict(zip( feat_names, model.feature_importances_))
    with open(os.path.join(root, 'data', 'models', 'model_%s_importance' % name), 'w') as fipt:
        for each in sorted(dFeatImps.iteritems(), key = lambda a: a[1], reverse=True):
            print >> fipt, each[0], ",", each[1]

def main(argv):
    conf_file = argv[0]
    pool_num = int(argv[1])


    importstr = "import %s as conf" % conf_file
    exec importstr

    import model.model_param_set as params_set


    pool = multiprocessing.Pool(processes=pool_num)
    result = []
    for name in conf.l_params:
        params = params_set.d_all[name]
        if pool_num <= 1:
            one_work(name, params[0], params[1], params[2], params[3])
        else:
            result.append(pool.apply_async(one_work, (name, params[0], params[1], params[2], params[3])))
    pool.close()
    pool.join()
    for each in result:
        print each.get()

if __name__ == '__main__':
    main(sys.argv[1:])
