#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os
import time
import numpy as np
import pandas as pd
from sklearn.externals import joblib # to dump model
import cPickle as pkl
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)
sys.path.append(local_path)

import model.modeling as  model
import model.model_param_set as model_params
from utils import time_me

def get_all_from(path):
    sym2df = {}
    for each in model.get_file_list(path):
        symbol = model.get_stock_from_path(each)
        df = pd.read_csv(each, parse_dates=True)
        df["sym"] = symbol
        sym2df[symbol] = df 
    print len(sym2df)
    return sym2df

def merge(sym2feats):
    dfMerged = None
    toAppends = []
    for sym in sym2feats.keys():
        df = sym2feats[sym]
        if dfMerged is None:
            dfMerged = df
        else:
            toAppends.append(df)
    # batch merge speeds up!
    dfMerged =  dfMerged.append(toAppends)
    return dfMerged

@time_me
def save(df, f):

    if df.shape[0] < 5000000:
        version = 0
    else:
        version = 0
    with open(f, 'wb') as ff:
        pkl.dump(df, ff,version)
    #df.to_csv(f)
def main(argv):
    for ta in model_params.d_dir_ta:
        sym2ta = get_all_from(model_params.d_dir_ta[ta])
        df = merge(sym2ta)
        print df.shape
        df = df.replace([np.inf,-np.inf],np.nan)\
            .dropna()
        print df.shape
        out_file = os.path.join(model_params.d_dir_ta[ta], "merged.pkl")
        #with open(out_file, 'w') as f:
        #    pkl.dump(df, f)
        #joblib.dump(df, out_file, compress = 3)
        save(df, out_file)
if __name__ == '__main__':
    main(sys.argv)



