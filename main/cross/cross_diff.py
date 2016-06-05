#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os
import json
import time
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.externals import joblib # to dump model
import cPickle as pkl
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)
sys.path.append(local_path)

import model.modeling as  model
from utils import time_me

def accu(df, label, threshold):
    npPred  = df["pred"].values
    npLabel = df[label].values
    npPos = npPred[npPred >= threshold]
    npTrueInPos = npLabel[(npPred >= threshold) & (npLabel>1.0)]
    npTrue = npLabel[npLabel > 1.0]
    return {"pos": npPos.size, "trueInPos":npTrueInPos.size}

@time_me
def get_df(f):
    with open(f, "rb") as ff:
        df = pkl.load(ff)
    #return joblib.load(f)
    #return pd.read_csv(f)
    return df

def get_range(df, start ,end):
    return df.query('date >="%s" & date <= "%s"' % (start, end)) 

def one_work(cls, ta_dir, label, date_range, th):
    re =  "%s\t%s\t%s\t%s\t%s\t%f\t" % (cls, ta_dir[-4:], label, date_range[0], date_range[1],th)
    merged_file = os.path.join(ta_dir, "merged.pkl")
    df = get_df(merged_file)
    df = get_range(df, date_range[0], date_range[1])
    cls = joblib.load(os.path.join(root, 'data', 'models',"model_" + cls + ".pkl"))
    feat_names = model.get_feat_names(df)
    npFeat = df.loc[:,feat_names].values
    npPred = cls.predict_proba(npFeat)[:,1]
    df["pred"] = npPred
    dacc =  accu(df, label, th)
    re += "%f\t%f\t%f" % (dacc["trueInPos"], dacc["pos"], dacc["trueInPos"]*1.0 / dacc["pos"])
    return re

def main(argv):
    pool_num = int(argv[1])
    conf_file = argv[2]
    impstr = "import %s as conf" % conf_file
    exec impstr
    out_file = os.path.join(root, 'data', "crosses", conf_file+".report")
    fout = open(out_file, 'w')


    pool = multiprocessing.Pool(processes=pool_num)
    result = []
    for each in conf.l_params:
        one_work(*each)
        #result.append(pool.apply_async(one_work, each ))
    pool.close()
    pool.join()
    for each in result:
        print >> fout, "%s" % each.get()
    fout.close()

if __name__ == '__main__':
    main(sys.argv)
    



