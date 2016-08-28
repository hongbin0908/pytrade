#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

"""
./pred/pred.py  call1s1_dow_GBCv1n322md3lr001_l5_s1700e2009 call1s1_dow 2016-01-01 2016-12-31  label5
"""

import sys,os
import json
import numpy as np
import pandas as pd
import math
import multiprocessing
import cPickle as pkl
from sklearn.externals import joblib # to dump model
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)
sys.path.append(local_path)

from main.utils import time_me
import main.base as base
import main.yeod.yeod as yeod

def main(args):
    lsym =  getattr(yeod, "get_%s" % args.setname)()
    if args.start is None:
        args.start = base.last_trade_date()
        args.end = args.start
    cls = joblib.load(os.path.join(base.dir_model(), args.model))
    
    ta = base.get_merged_with_na(args.taname, lsym)

    ta = ta[(ta['date'] >= args.start) & (ta['date'] <= args.end)]
    dfFeat = ta.loc[:, base.get_feat_names(ta)]
    dfFeat = dfFeat.replace([np.inf,-np.inf],np.nan)\
        .dropna()
    npFeat = dfFeat.values
    npPred = cls.predict_proba(npFeat)
    #for i, npPred in enumerate(cls.staged_predict_proba(npFeat)):
    #    if i == args.stage:
    #        break
    ta["pred"] = npPred[:,1]
    ta.sort("pred", inplace = True, ascending = False)
    freport, fcsv = base.file_pred(args)
    ta.to_csv(fcsv)
    #ta[["date", "sym", "pred", label]].to_csv(os.path.join(out_dir, 'pred.s.csv'))
    with open(freport, 'w') as fout:
        print >>fout,  ta[["date","sym", "pred"]].head(10)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='paper test')
    parser.add_argument('--stage', dest="stage", action="store", default=600, \
            help="the stage of gbdt")
    parser.add_argument('--start', dest="start", action="store", default=None)
    parser.add_argument('--end', dest="end", action="store", default=None)
    parser.add_argument('--label', dest="label", action="store", default="label5")
    parser.add_argument('model', help = "the full path of model")
    parser.add_argument('setname', help = "")
    parser.add_argument('taname', help = "")

    args = parser.parse_args()
    main(args)
