#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os
import json
import numpy as np
import pandas as pd
from sklearn.externals import joblib # to dump model
import cPickle as pkl
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)
sys.path.append(local_path)

import model.modeling as  model

def accu(df, label, threshold):
    npPred  = df["pred"].values
    npLabel = df[label].values
    npPos = npPred[npPred >= threshold]
    npTrueInPos = npLabel[(npPred >= threshold) & (npLabel>1.0)]
    npTrue = npLabel[npLabel > 1.0]
    return {"pos": npPos.size, "trueInPos":npTrueInPos.size}

def main(argv):
    conf_file = argv[1]
    impstr = "import %s as conf" % conf_file
    exec impstr

    dfAll = None
    sym2ta = None
    for each in conf.l_params:
        print each
        with open(os.path.join(each[1], "merged.pkl")) as f:
            df = pkl.load(f)
        df.sort_index()
        cls = joblib.load(os.path.join(root, 'data', 'models',"model_" + each[0]+ ".pkl"))
        df = df.query('date >="%s" & date <= "%s"' % (each[3][0], each[3][1])) 
        feat_names = model.get_feat_names(df)
        npFeat = df.loc[:,feat_names].values
        npPred = cls.predict_proba(npFeat)[:,1]
        df["pred"] = npPred
        dacc =  accu(df, each[2], each[4])
        print dacc["trueInPos"], dacc["pos"], dacc["trueInPos"]*1.0 / dacc["pos"]

if __name__ == '__main__':
    main(sys.argv)
    



