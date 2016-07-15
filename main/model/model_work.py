#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

import sys,os
import json
import numpy as np
import pandas as pd
import math
import random
import multiprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib # to dump model
from sklearn import preprocessing

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

import main.model.model_param_set as params_set
import main.base as base
import main.ta as ta

isscaler = False

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

def getMinMax(npTrainFeat):
    minMaxScaler = preprocessing.MinMaxScaler()
    npTrainFeatMinMax = minMaxScaler.fit_transform(npTrainFeat)
    return minMaxScaler

def one_work(taName, model, label, start, end, isrepeat, sample):
    dir_ta = base.dir_ta(taName)
    cls = params_set.d_model[model]
    if isrepeat == 1:
        if sample > 0:
            name = "model_%s_%s_%s_%s_%s_%s.re.pkl" % (taName,model,label, start, end,sample)
            scalerName = "scaler_%s_%s_%s_%s_%s_%s.re.pkl" % (taName,model,label, start, end,sample)
        else:
            name = "model_%s_%s_%s_%s_%s.re.pkl" % (taName,model,label, start, end)
            scalerName = "scaler_%s_%s_%s_%s_%s.re.pkl" % (taName,model,label, start, end)
    elif isrepeat == 2:
        if sample > 0:
            name = "model_%s_%s_%s_%s_%s_%s.se.pkl" % (taName,model,label, start, end,sample)
            scalerName = "scaler_%s_%s_%s_%s_%s_%s.se.pkl" % (taName,model,label, start, end,sample)
        else:
            name = "model_%s_%s_%s_%s_%s.re.pkl" % (taName,model,label, start, end)
            scalerName = "scaler_%s_%s_%s_%s_%s.re.pkl" % (taName,model,label, start, end)
    else:
        if sample > 0:
            name = "model_%s_%s_%s_%s_%s_%s.pkl" % (taName,model,label, start, end,sample)
            scalerName = "scaler_%s_%s_%s_%s_%s_%s.pkl" % (taName,model,label, start, end,sample)
        else:
            name = "model_%s_%s_%s_%s_%s.pkl" % (taName,model,label, start, end)
            scalerName = "scaler_%s_%s_%s_%s_%s.pkl" % (taName,model,label, start, end)

    if os.path.isfile(os.path.join(root, 'data', 'models',  name)):
        print "%s already exists!" % name
        return
    dfTa = base.get_merged(dir_ta)
    dfTrain = build_trains(dfTa, start, end)

    if sample > 0:
        sample = len(dfTrain)/sample
        rows = random.sample(range(len(dfTrain)), sample)
        print "xx", dfTrain.shape
        print len(rows)
        dfTrain = dfTrain.reset_index(drop=True)
        dfTrain = dfTrain.ix[rows]
        print "xx", dfTrain.shape
    
    if isrepeat ==1 :
        toAppends = []
        for i in range(1,10):
            dfTmp = dfTrain[dfTrain.label5>=1+i/100.0]
            toAppends.append(dfTmp)
        print dfTrain.shape
        dfTrain = dfTrain.append(toAppends)
        print dfTrain.shape
    elif isrepeat == 2:
        print dfTrain.shape
        dfTrain = dfTrain[abs(dfTrain.label5 -1) > 0.01]
        print dfTrain.shape


    feat_names = base.get_feat_names(dfTrain)
    npTrainFeat = dfTrain.loc[:,feat_names].values

    npTrainLabel = dfTrain.loc[:,label].values.copy()
    npTrainLabel[npTrainLabel != 1.0]
    npTrainLabel[npTrainLabel >  1.0] = 1
    npTrainLabel[npTrainLabel <  1.0] = 0

    if isscaler:
        scaler = getMinMax(npTrainFeat)
        npTrainFeatScaled = scaler.transform(npTrainFeat)
    else:
        npTrainFeatScaled = npTrainFeat
    cls.fit(npTrainFeatScaled, npTrainLabel)
    joblib.dump(cls, os.path.join(root, "data", "models", name), compress = 3)
    joblib.dump(scaler, os.path.join(root, 'data', 'models',scalerName), compress = 3)
    dFeatImps = dict(zip( feat_names, cls.feature_importances_))
    with open(os.path.join(root, 'data', 'models', '%s_importance' % name), 'w') as fipt:
        for each in sorted(dFeatImps.iteritems(), key = lambda a: a[1], reverse=True):
            print >> fipt, each[0], ",", each[1]

def main(argv):
    taName = argv[0]
    clsName = argv[1]
    labelName = argv[2]
    start = argv[3]
    end = argv[4]
    isrepeat = int(argv[5])
    sample = int(argv[6])
    one_work(taName, clsName, labelName, start, end, isrepeat, sample)
if __name__ == '__main__':
    main(sys.argv[1:])
