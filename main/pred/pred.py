#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@author  Bin Hong

import sys,os
import json
import numpy as np
import pandas as pd
import math
import multiprocessing
from sklearn.externals import joblib # to dump model
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)
sys.path.append(local_path)
def get_cls(clsName):
    return joblib.load(os.path.join(root, 'data', 'models', "model_" + clsName + ".pkl"))
def get_ta(taName):
    return pkl.load(os.path.join(root, 'data', taName, 'merged_with_na.pkl')) 
def main(argv):
    clsName = argv[1]
    taName = argv[2]
    date_ = argv[3]

    cls = get_cls(clsName)
    ta = get_ta(taName)
    ta = ta.query('date == %s' % date_)
    npFeat = ta.loc[:, model.get_feat_names(ta)].values
    npPred = cls.predict_proba(npFeat)[:,1]
    ta["pred"] = npPred
    ta.to_csv(os.path.join(root, 'data', 'pred', 'pred_' + clsName + "_" + taName + "_" + date_ + ".csv"))
if __name__ == '__main__':
    main(sys.argv)
