#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

"""
"""

import sys,os
import json
import numpy as np
import pandas as pd
from sklearn.externals import joblib # to dump model

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

import main.base as base
import main.ta as ta
import main.paper.ana as ana
import main.yeod.yeod as yeod

isscaler = False


def get_scaler(clsName):
    return joblib.load(os.path.join(root, 'data', 'models', 'scaler_' + clsName + ".pkl"))


def main(args):
    lsym =  getattr(yeod, "get_%s" % args.setname)()
    dfTa = base.get_merged(args.taname, lsym)
    if dfTa is None:
        print "can not merge " % args.setname
        sys.exit(1)
    dfTa = base.get_range(dfTa, args.start, args.end)
    print dfTa.shape
    #if args.filter:
    #    dfTa = filter_trend(dfTa)
    #print dfTa.shape

    cls = joblib.load(os.path.join(base.dir_model(),args.model))
    feat_names = base.get_feat_names(dfTa)
    npFeat = dfTa.loc[:,feat_names].values
    if isscaler :
        scaler = get_scaler(clsName)
        npFeatScaler = scaler.transform(npFeat)
    else:
        npFeatScaler = npFeat
    #for i, npPred in enumerate(cls.staged_predict_proba(npFeatScaler)):
    #    if i == args.stage:
    #        break
    npPred = cls.predict_proba(npFeat)
    dfTa["pred"] = npPred[:,1]
    dfTa = dfTa.sort_values(['pred'], ascending = False)
    freport,fpred = base.file_paper(args)
    dfTa.to_csv(fpred)

    ana.main([fpred, args.top, args.thresh,freport, args.level])
    print freport
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='paper test')
    parser.add_argument('--stage', dest="stage", action="store", default=600, \
            help="the stage of gbdt")
    parser.add_argument('--start', dest="start", action="store", default='2010-01-01')
    parser.add_argument('--end', dest="end", action="store", default='2016-12-31')
    parser.add_argument('--top', dest="top", action="store", type = int, default=2)
    parser.add_argument('--thresh', dest="thresh", action="store", type = int, default=800)
    parser.add_argument('--level', dest="level", action="store", type=float, default=1.0)
    parser.add_argument('model', help = "the full path of model")
    parser.add_argument('setname', help = "")
    parser.add_argument('taname', help = "")

    args = parser.parse_args()
    main(args)
