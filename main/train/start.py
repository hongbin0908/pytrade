#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong
import sys,os
import json
import numpy as np
import multiprocessing
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)
from model_conf import d_conf
from models import d_choris
from model_base import *
from sklearn.ensemble import GradientBoostingClassifier

def get_feat_names(df):
    return [x for x in df.columns if x.startswith('ta_')]
def get_label_name(df, level):
    return "label" + str(level)


def build_trains(sym2feats, start, end, isDropNa = True):
    if isDropNa:
        dfTrains = merge(sym2feats, start ,end).replace([np.inf,-np.inf],np.nan).dropna()
    else:
        dfTrains = merge(sym2feats, start ,end)
    return dfTrains

def build_preds(sym2feats, start, end):
    df = merge(sym2feats, start, end)
    #print df.shape
    for x in df.columns:
        if x.startswith('label'):
            del df[x]
    del df['close_shift']
    #print df.shape
    #print df.iloc[1]
    df = df.replace([np.inf,-np.inf],np.nan).dropna()
    print df.shape
    return df


def merge(sym2feats, start ,end):
    dfMerged = None
    toAppends = []
    for sym in sym2feats.keys():
        df = sym2feats[sym]
        df = df.loc[start:end,:]
        #assert False == df.head().isnull().any(axis=1).values[0] 
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
            #if len(toAppends) > 5:
            #    break
    dfMerged =  dfMerged.append(toAppends)
    return dfMerged


def pred(sym2feats, level, params, start1, end1, start2, end2):
    dfTrain = build_trains(sym2feats, start1, end1)
    dfTest = build_preds(sym2feats, start2, end2)
    model = GradientBoostingClassifier(**params)
    npTrainFeat = dfTrain.loc[:,get_feat_names(dfTrain)].values
    npTrainLabel = dfTrain.loc[:,get_label_name(dfTrain,level)].values
    fil = npTrainLabel.copy()
    npTrainLabel[fil>= 1.0] = 1
    npTrainLabel[fil<  1.0] = 0
    model.fit(npTrainFeat, npTrainLabel)

    npTestFeat = dfTest.loc[:,get_feat_names(dfTrain)].values

    dfTest["pred"] = model.predict_proba(npTestFeat)[:,1]
    return dfTest

def train2(sym2feats, level, params, start1, end1, start2, end2):
    dfTrain = build_trains(sym2feats, start1, end1)
    dfTest = build_trains(sym2feats, start2, end2)
    model = GradientBoostingClassifier(**params)
    feat_names = get_feat_names(dfTrain)
    npTrainFeat = dfTrain.loc[:,feat_names].values
    npTrainLabel = dfTrain.loc[:,get_label_name(dfTrain,level)].values.copy()
    print npTrainLabel.shape
    npTrainLabel[npTrainLabel != 1.0]
    print npTrainLabel.shape
    npTrainLabel[npTrainLabel >  1.0] = 1
    npTrainLabel[npTrainLabel <  1.0] = 0
    model.fit(npTrainFeat, npTrainLabel)

    npTestFeat = dfTest.loc[:,feat_names].values
    #npTestLabel = dfTest.loc[:,get_label_name(dfTrain,level)].values.copy()
    #print npTestLabel.shape
    #npTestLabel[npTestLabel != 1.0]
    #print npTestLabel.shape
    #npTestLabel[npTestLabel> 1.0] = 1
    #npTestLabel[npTestLabel<= 1.0] = 0

    dfTest["pred"] = model.predict_proba(npTestFeat)[:,1]
    dFeatImps = dict(zip( feat_names, model.feature_importances_))
    for each in sorted(dFeatImps.iteritems(), key = lambda a: a[1], reverse=True):
            print each
     
    return dfTest

def get_stock_data_pd(path):
    df = pd.read_csv(path,  index_col = 'date', parse_dates=True).sort_index()
    return df

def get_all_from(path):
    sym2df = {}
    for each in get_file_list(path):
        symbol = get_stock_from_path(each)
        df = get_stock_data_pd(each)
        sym2df[symbol] = df 
    return sym2df


def one_work(idx, path, level, params, range_):
    print idx, path, level, params, range_
    dir_pred = os.path.join(local_path, '..', 'data', 'pred', str(idx))
    if not os.path.isdir(dir_pred):
        os.mkdir(dir_pred)

    with open(os.path.join(dir_pred, 'desc'), 'w') as fdesc:
        ddesc = {"ta":path, "level":level, "params":params}
        print >> fdesc, "%s" % (json.dumps(ddesc))
    sym2feats = get_all_from(path)
    pred_start, pred_end = range_[0]
    ltestrange = range_[1]
    #print "======PREDING %s ========" % str((pred_start, pred_end))
    #dfTest = pred(sym2feats, level, params, pred_start, pred_end, '2016-05-27', '2016-05-28')
    #print dfTest.shape
    #dfTest.to_csv(os.path.join(dir_pred, "today_%s.csv" % "2016-05-27"))
    dfTestAll = None
    for each in ltestrange:
        print "====== TRAING %s =====" % str(each)
        dfTest = train2(sym2feats, level, params, each[0], each[1], each[2], each[3]); 
        if dfTestAll is None:
            dfTestAll = dfTest
        else:
            dfTestAll = dfTestAll.append(dfTest)
    dfTestAll.to_csv(os.path.join(dir_pred, 'pred.csv'))

def main(argv):
    pool_num = int(argv[1])
    str_conf = argv[2]
    l_conf = d_conf[str_conf]
    pool = multiprocessing.Pool(processes=pool_num)
    result = []
    for each in l_conf:
        m = d_choris[each]
        params = (each, m[0], m[1], m[2],m[3])
        result.append(pool.apply_async(one_work, params))
        #one_work(params[0], params[1], params[2], params[3], params[4])
    pool.close()
    pool.join()
    for each in result:
        print each.get()

if __name__ == '__main__':
    main(sys.argv)
