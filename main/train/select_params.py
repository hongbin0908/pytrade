#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys, os
import json
import math
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)


import train.start as start

def main(argv):
    sym_dir = os.path.join(root, 'data', 'ta1')
    symToTa = start.get_all_from(sym_dir)
    dfTrain = start.build_trains(symToTa, '1970-01-01', '2099-12-31')
    print dfTrain.shape
    feat_names = start.get_feat_names(dfTrain)
    npTrainFeat = dfTrain.loc[:,feat_names].values
    npTrainLabel = dfTrain.loc[:,start.get_label_name(dfTrain,3)].values.copy()
    npTrainLabel[npTrainLabel != 1.0]
    npTrainLabel[npTrainLabel >  1.0] = 1
    npTrainLabel[npTrainLabel <  1.0] = 0
    param_grid = {'learning_rate':[0.1, 0.01],
                      'max_depth':[3,6],
                      'min_samples_leaf':[1,5,17],
                      'max_features':[1.0,  0.5]}
    est = GradientBoostingClassifier(verbose = 1, n_estimators = 1000)
    gs_cv = GridSearchCV(est, param_grid,n_jobs=8, verbose=1 ).fit(npTrainFeat, npTrainLabel)
    print gs_cv.best_params_
if __name__ == '__main__':
    main(sys.argv)
