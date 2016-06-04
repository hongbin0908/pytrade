#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os
import json
import numpy as np
import pandas as pd
import math
import multiprocessing
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)
sys.path.append(local_path)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib # to dump model

def get_file_list(rootdir):
    file_list = []
    for f in os.listdir(rootdir):
        if f == None or not f.endswith(".csv"):
            continue
        file_list.append(os.path.join(rootdir, f))
    return file_list

def get_stock_from_path(pathname):
    return os.path.splitext(os.path.split(pathname)[-1])[0]

def get_feat_names(df):
    """
    the the columns of features name to train 
    """
    return [x for x in df.columns if x.startswith('ta_')]

def get_stock_data_pd(path):
    df = pd.read_csv(path,  index_col = 'date', parse_dates=True).sort_index()
    return df

def get_all_from(path):
    sym2df = {}
    for each in get_file_list(path):
        symbol = get_stock_from_path(each)
        df = get_stock_data_pd(each)
        mean =  np.max(np.abs((df.tail(11)["label1"].head(10).values - 1)))
        if mean < 0.01:
            print symbol, mean
            continue
        if np.max(df.tail(11)["volume"].head(10).values) < 500000:
            print 2, symbol, mean
            continue
        sym2df[symbol] = df 
    print len(sym2df)
    return sym2df

def build_trains(sym2feats, start, end):
    """
    param sym2feats:
        dict from symbol to its features dataframe
    param start:
        the first day(include) to merge
    param end:
        the last day(exclude) to merge
    """
    # sometimes there is inf value in features
    df = merge(sym2feats, start ,end)\
            .replace([np.inf,-np.inf],np.nan)\
            .dropna()
    return df

def merge(sym2feats, start ,end):
    """
    merge the features dataframe and add [sym, date] index.
    **both** the start and the stop are included!
    """
    dfMerged = None
    toAppends = []
    for sym in sym2feats.keys():
        df = sym2feats[sym]
        df = df.loc[start:end,:]
        index2 = df.index.values
        index1 = []
        for i in range(0,df.shape[0]):
            index1.append(sym)
        df = pd.DataFrame(df.values, index = [index1, index2], columns = df.columns.values )
        df.index.names = ['sym','date']
        if dfMerged is None:
            dfMerged = df
        else:
            toAppends.append(df)
    # batch merge speeds up!
    dfMerged =  dfMerged.append(toAppends)
    return dfMerged


def one_work(name, dir_ta, model, label, date_range):
    if os.path.isfile(os.path.join(root, 'data', 'models', "model_%s.pkl" % name)):
        print "%s already exists!" % name
        return
    sym2ta = get_all_from(dir_ta)
    dfTrain = build_trains(sym2ta, date_range[0], date_range[1])
    feat_names = get_feat_names(dfTrain)
    npTrainFeat = dfTrain.loc[:,feat_names].values
    npTrainLabel = dfTrain.loc[:,label].values.copy()
    npTrainLabel[npTrainLabel != 1.0]
    npTrainLabel[npTrainLabel >  1.0] = 1
    npTrainLabel[npTrainLabel <  1.0] = 0
    model.fit(npTrainFeat, npTrainLabel)
    joblib.dump(model, os.path.join(root, "data", "models", "model_%s.pkl" % name), compress = 3)
    dFeatImps = dict(zip( feat_names, model.feature_importances_))
    with open(os.path.join(root, 'data', 'models', 'model_%s_importance' % name), 'w') as fipt:
        for each in sorted(dFeatImps.iteritems(), key = lambda a: a[1], reverse=True):
            print >> fipt, each[0], ",", each[1]

def main(argv):
    pool_num = int(argv[1])
    conf_file = argv[2]
    importstr = "import %s as conf" % conf_file
    import train.model_param_set as params_set
    exec importstr

    pool = multiprocessing.Pool(processes=pool_num)
    result = []
    for name in conf.l_params:
        params = params_set.d_all[name]
        result.append(pool.apply_async(one_work, (name, params[0], params[1], params[2], params[3])))
    pool.close()
    pool.join()
    for each in result:
        print each.get()

if __name__ == '__main__':
    main(sys.argv)
