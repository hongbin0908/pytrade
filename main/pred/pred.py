#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@author  Bin Hong

import sys,os
import json
import numpy as np
import pandas as pd
import math
import multiprocessing
import cPickle as pkl
from sklearn.externals import joblib # to dump model
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)
sys.path.append(local_path)
import model.modeling as  model
from utils import time_me
def get_cls(clsName):
    return joblib.load(os.path.join(root, 'data', 'models', "model_" + clsName + ".pkl"))
@time_me
def get_ta(taName):
    return pd.read_hdf(os.path.join(root, 'data', taName, 'merged_wth_na.pkl'), "df")
    return pkl.load(open(os.path.join(root, 'data', taName, 'merged_wth_na.pkl'))) 
def main(argv):
    clsName = argv[1]
    taName = argv[2]
    date_ = argv[3]

    cls = get_cls(clsName)
    ta = get_ta(taName)
    ta = ta.query('date == "%s"' % date_)
    np.set_printoptions(threshold='nan')
    print ta[['sym', 'date','ta_adx_14']].values

    npFeat = ta.loc[:, model.get_feat_names(ta)].values
    npPred = cls.predict_proba(npFeat)[:,1]
    ta["pred"] = npPred
    ta.sort_values("pred", inplace = True, ascending = False)
    ta.to_csv(os.path.join(root, 'data', 'pred', 'pred_' + clsName + "_" + taName + "_" + date_ + ".csv"))
if __name__ == '__main__':
    main(sys.argv)
