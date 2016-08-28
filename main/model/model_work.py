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
import main.model.sample_weight as sw
import main.yeod.yeod as yeod


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

def main(args):
    cls = params_set.d_model[args.clsname]
    file_model, file_ipt = base.file_model(args)

    if os.path.isfile(file_model):
        print "%s already exists!" % file_model
        return
    dfTa = base.get_merged(args.taname, getattr(yeod, "get_%s" % args.setname)())
    if dfTa is None:
        return None
    dfTrain = build_trains(dfTa, args.start, args.end)

    if args.sample:
        print "sampling ..."
        sample = len(dfTrain)/sample
        rows = random.sample(range(len(dfTrain)), sample)
        print len(rows)
        dfTrain = dfTrain.reset_index(drop=True)
        dfTrain = dfTrain.ix[rows]
    
    if args.repeat:
        print "repeat ..."
        toAppends = []
        for i in range(1,3):
            dfTmp = dfTrain[dfTrain.label5>=1+i/20.0]
            toAppends.append(dfTmp)
        print dfTrain.shape
        dfTrain = dfTrain.append(toAppends)
        print dfTrain.shape

    if args.sw:
        dfTrain = getattr(sw, "sw_%s" % args.sw)(dfTrain)

    feat_names = base.get_feat_names(dfTrain)
    npTrainFeat = dfTrain.loc[:,feat_names].values

    npTrainLabel = dfTrain.loc[:,args.labelname].values.copy()
    npTrainLabel[npTrainLabel != 1.0]
    npTrainLabel[npTrainLabel >  1.0] = 1
    npTrainLabel[npTrainLabel <  1.0] = 0

    if args.scaler:
        scaler = getMinMax(npTrainFeat)
        npTrainFeatScaled = scaler.transform(npTrainFeat)
    else:
        npTrainFeatScaled = npTrainFeat
    if args.sw:
        cls.fit(npTrainFeatScaled, npTrainLabel, sample_weight=dfTrain["sample_weight"].values)
    else:
        cls.fit(npTrainFeatScaled, npTrainLabel)
    joblib.dump(cls, file_model, compress = 3)
    #joblib.dump(scaler, os.path.join(root, 'data', 'models',scalerName), compress = 3)
    dFeatImps = dict(zip( feat_names, cls.feature_importances_))
    with open(file_ipt, 'w') as fipt:
        for each in sorted(dFeatImps.iteritems(), key = lambda a: a[1], reverse=True):
            print >> fipt, each[0], ",", each[1]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='traing modeling ')
    parser.add_argument('-s', '--sw', help='is sample weight', dest='sw', action="store", default = None)
    parser.add_argument('-r', '--rp', help="repeat",    dest="repeat", action="store", default=None)
    parser.add_argument('--sample', help="sample method", dest="sample", action="store", default=None)
    parser.add_argument('--scaler', help="scaler method", dest="scaler", action="store", default=None)
    parser.add_argument('--start', dest='start', action='store', default='1700-01-01', help="model start time")
    parser.add_argument('--end',   dest='end',   action='store', default='2009-12-31', help="model end time")
    parser.add_argument('--label', dest='labelname', action='store', default='label5', help="the label name")
    
    parser.add_argument('setname', help = "the sym set to be ta")
    parser.add_argument('taname', help = "the sym set to be ta")
    parser.add_argument('clsname', help="the model full name")

    args = parser.parse_args()
    main(args)
