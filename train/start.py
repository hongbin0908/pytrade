
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author 
import sys,os
import numpy as np
import multiprocessing
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)
from model_base import *
from sklearn import metrics
from sklearn import linear_model
from sklearn import tree
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import neural_network
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor 
from sklearn import preprocessing

def get_feat_names(df):
    #return ["ta_adx_14", "ta_mdi_14", "ta_pdi_14", "ta_ad"]
    #return ["ta_adx_14","ta_adxr_14","ta_mdi_14","ta_pdi_14",\
    #        "ta_apo_12_26_0","ta_aroon_up_14","ta_aroon_down_14",\
    #        "ta_ad",\
    #        "diff_0_1","diff_1_1","diff_2_1","diff_3_1",\
    #        "ta_aroonosc_14",\
    #        "ta_adsoc","ta_obv","ta_atr_14","ta_natr_14","ta_trange"]
    return [x for x in df.columns if x.startswith('ta_')]
def get_label_name(df, level):
    return "label" + str(level)


def build_trains(sym2feats, start, end, isDropNa = True):
    if isDropNa:
        dfTrains = merge(sym2feats, start ,end).dropna()
    else:
        dfTrains = merge(sym2feats, start ,end)
    return dfTrains


def ana(npTestLabel, npPred, threshold):
    npPos = npPred[npPred >= 1+threshold]
    npTrueInPos = npTestLabel[(npPred >= 1.0+threshold) & (npTestLabel>=1.0)]
    npNeg = npPred[npPred < 1-threshold]
    npFalseInNeg = npTestLabel[(npPred < 1.0-threshold) & (npTestLabel< 1.0)]
    npTrue = npTestLabel[npTestLabel >= 1.0]
    print "%d\t%d\t" % (npPos.size, npTrueInPos.size),
    if npPos.size > 0:
        print npTrueInPos.size*1.0/npPos.size,
    else:
        print 0.0,
    print "%d\t%d\t" % (npNeg.size, npFalseInNeg.size),
    if npNeg.size > 0:
        print npFalseInNeg.size*1.0/npNeg.size,
    else:
        print 0.0,
    print "%d\t%d\t%f\t" % (npTrue.size, npTestLabel.size, npTrue.size*1.0/npTestLabel.size)

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

def cal_cor(df, feat,pos, neg, label):
    npFeat = df[feat].values
    npLabel = df[label].values
    npIsPos = (npFeat >= pos)
    npIsNeg = (npFeat < neg)
    npIsTrue = (npLabel >= 1.0)
    npIsFalse = (npLabel < 1.0)
    dfTrueInPos = df[npIsPos & npIsTrue]
    dfPos = df[npIsPos]
    dfFalseInNeg = df[npIsNeg & npIsFalse]
    dfNeg = df[npIsNeg]
    dfTrue = df[npIsTrue]
    dfFalse = df[npIsFalse]
    print feat, label, 
    print dfPos.shape[0], dfTrueInPos.shape[0], 
    if dfPos.shape[0]> 0:
        print dfTrueInPos.shape[0]*1.0/dfPos.shape[0], 
    else:
        print 0.0,
    print dfTrue.shape[0]*1.0/df.shape[0],
    print dfNeg.shape[0], dfFalseInNeg.shape[0], 
    if dfNeg.shape[0] > 0:
        print dfFalseInNeg.shape[0]*1.0/dfNeg.shape[0], 
    else:
        print 0.0,
    print dfFalse.shape[0]*1.0/df.shape[0],
    print 

def ana_cls(dfTest, level, threshold):
    print level,threshold,
    dfPos = dfTest[dfTest["pred"] >= threshold]
    dfTrueInPos = dfPos[dfPos[get_label_name(dfPos, level)]==1.0]
    print dfPos.shape[0],
    if dfPos.shape[0] > 0:
        print dfTrueInPos.shape[0]*1.0/dfPos.shape[0]
    else:
        print 0.0

def pred(sym2feats, level, params, start1, end1, start2, end2):
    dfTrain = build_trains(sym2feats, start1, end1)
    dfTest = build_trains(sym2feats, start2, end2, isDropNa = False)
    model = GradientBoostingClassifier(**params)
    npTrainFeat = dfTrain.loc[:,get_feat_names(dfTrain)].values
    npTrainLabel = dfTrain.loc[:,get_label_name(dfTrain,level)].values
    fil = npTrainLabel.copy()
    npTrainLabel[fil>= 1.0] = 1
    npTrainLabel[fil<  1.0] = 0
    model.fit(npTrainFeat, npTrainLabel)

    npTestFeat = dfTest.loc[:,get_feat_names(dfTrain)].values
    npTestLabel = dfTest.loc[:,get_label_name(dfTrain,level)].values
    npTestLabel[npTestLabel>=1.0] = 1
    npTestLabel[npTestLabel<1.0] = 0

    dfTest["pred"] = model.predict_proba(npTestFeat)[:,1]
    return dfTest

def train2(sym2feats, level, params, start1, end1, start2, end2):
    dfTrain = build_trains(sym2feats, start1, end1)
    dfTest = build_trains(sym2feats, start2, end2)
    # Fit classifier with out-of-bag estimates
    #params = {'verbose':1,'n_estimators': 40, 'max_depth': 3, 'subsample': 0.5,
    #                  'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
    #params = {'verbose':0,'n_estimators':500, 'max_depth':4 }
    model = GradientBoostingClassifier(**params)
    npTrainFeat = dfTrain.loc[:,get_feat_names(dfTrain)].values
    npTrainLabel = dfTrain.loc[:,get_label_name(dfTrain,level)].values
    fil = npTrainLabel.copy()
    npTrainLabel[fil>= 1.0] = 1
    npTrainLabel[fil<  1.0] = 0
    model.fit(npTrainFeat, npTrainLabel)

    npTestFeat = dfTest.loc[:,get_feat_names(dfTrain)].values
    npTestLabel = dfTest.loc[:,get_label_name(dfTrain,level)].values
    npTestLabel[npTestLabel>=1.0] = 1
    npTestLabel[npTestLabel<1.0] = 0

    dfTest["pred"] = model.predict_proba(npTestFeat)[:,1]
    acc = model.score(npTestFeat, npTestLabel)
    print("Accuracy: {:.4f}".format(acc))
    ana_cls(dfTest,level, 0.5)
    ana_cls(dfTest,level, 0.55)
    ana_cls(dfTest,level, 0.6)
    ana_cls(dfTest,level, 0.65)
    ana_cls(dfTest,level, 0.7)
    ana_cls(dfTest,level, 0.8)
    #print metrics.mean_absolute_error(dfTest[get_label_name(dfTest,level)].values, dfTest["pred"].values)
    #dfTrain["pred"] = model.predict(dfTrain.loc[:,get_feat_names(dfTrain)].values)
    #print metrics.mean_absolute_error(dfTrain[get_label_name(dfTrain,level)].values, dfTrain["pred"].values)

    #test_score = np.zeros((n_estimators,),dtype=np.float64)
    #for i , y_pred in enumerate(model.staged_predict(dfTest[get_feat_names(dfTest)].values)):
    #    test_score[i] = model.loss_(dfTest[get_label_name(dfTest, level)].values, y_pred)
    #for i in range(n_estimators):
    #    print model.train_score_[i], test_score[i]

    #r2_score = metrics.r2_score(npTestLabel, npPred)
    #mse = metrics.mean_squared_error(npTestLabel, npPred)
    #print "r2_score:", r2_score, mse
    #ana(npTestLabel, npPred, 0.0)
    #ana(npTestLabel, npPred,0.005)
    #ana(npTestLabel, npPred, 0.01)
    #ana(npTestLabel, npPred, 0.02)
    #ana(npTestLabel, npPred, 0.03)
    #ana(npTestLabel, npPred, 0.04)
    #ana(npTestLabel, npPred, 0.06)
    #ana(npTestLabel, npPred, 0.07)
    return dfTest

def train(sym2feats, level, start1, end1, start2, end2):
    dfTrain = build_trains(sym2feats, start1, end1)
    dfTest = build_trains(sym2feats, start2, end2)
    n_estimators = 40
    model = GradientBoostingRegressor(loss = 'ls', n_estimators = n_estimators,\
            min_samples_split=1,learning_rate=0.1, max_depth=3, verbose=1)
    model.fit(dfTrain.loc[:,get_feat_names(dfTrain)].values, dfTrain.loc[:,get_label_name(dfTrain,level)].values)
    dfTest["pred"] = model.predict(dfTest.loc[:,get_feat_names(dfTrain)].values)
    print metrics.mean_absolute_error(dfTest[get_label_name(dfTest,level)].values, dfTest["pred"].values)
    dfTrain["pred"] = model.predict(dfTrain.loc[:,get_feat_names(dfTrain)].values)
    print metrics.mean_absolute_error(dfTrain[get_label_name(dfTrain,level)].values, dfTrain["pred"].values)

    test_score = np.zeros((n_estimators,),dtype=np.float64)
    for i , y_pred in enumerate(model.staged_predict(dfTest[get_feat_names(dfTest)].values)):
        test_score[i] = model.loss_(dfTest[get_label_name(dfTest, level)].values, y_pred)
    for i in range(n_estimators):
        print model.train_score_[i], test_score[i]

    #r2_score = metrics.r2_score(npTestLabel, npPred)
    #mse = metrics.mean_squared_error(npTestLabel, npPred)
    #print "r2_score:", r2_score, mse
    #ana(npTestLabel, npPred, 0.0)
    #ana(npTestLabel, npPred,0.005)
    #ana(npTestLabel, npPred, 0.01)
    #ana(npTestLabel, npPred, 0.02)
    #ana(npTestLabel, npPred, 0.03)
    #ana(npTestLabel, npPred, 0.04)
    #ana(npTestLabel, npPred, 0.06)
    #ana(npTestLabel, npPred, 0.07)
    return dfTest#
def get_stock_data_pd(path):
    df = pd.read_csv(path,  index_col = 'date', parse_dates=True).sort_index()
    return df

def get_all_from(path):
    sym2df = {}
    i = 0
    for each in get_file_list(path):
        symbol = get_stock_from_path(each)
        df = get_stock_data_pd(each)
        sym2df[symbol] = df #.dropna()
        i += 1
    return sym2df

def one_work(idx, path, level, params):
    dir_pred = os.path.join(local_path, '..', 'data', 'pred', str(idx))
    if not os.path.isdir(dir_pred):
        os.mkdir(dir_pred)

    with open(os.path.join(dir_pred, 'desc'), 'w') as fdesc:
        print >> fdesc, "%s,%s,%s" % (str(path), str(level), str(params))
    sym2feats = get_all_from(path)
    dfTest = pred(sym2feats, level, params, '2006-01-1', '2016-01-01', '2016-05-18', '2016-05-19')
    dfTest.to_csv(os.path.join(dir_pred, "today_%s.csv" % "2016-05-18"))
    dfTest = train2(sym2feats, level, params, '2004-12-01', '2014-01-01', '2004-01-01', '2005-01-01'); dfTestAll = dfTest
    dfTest = train2(sym2feats, level, params, '2003-01-01', '2013-01-01', '2013-01-01', '2014-01-01'); dfTestAll = dfTestAll.append(dfTest)
    dfTest = train2(sym2feats, level, params, '2002-01-01', '2012-01-01', '2012-01-01', '2013-01-01'); dfTestAll = dfTestAll.append(dfTest)
    dfTestAll.to_csv(os.path.join(dir_pred, 'pred.csv'))


def main():
    datas = [
        os.path.join(local_path, '..', 'data','ta1'),
        os.path.join(local_path, '..', 'data','ta2')
    ]
    lParams = [
        {'verbose':1,'n_estimators':500, 'max_depth':3},                            #0
        {'verbose':1,'n_estimators':50, 'max_depth':5},
        {'verbose':0,'n_estimators':500, 'max_depth':4},
        {'verbose':0,'n_estimators':500, 'max_depth':4},
        {'verbose':0,'n_estimators':500, 'max_depth':4, 'learning_rate':0.01},
        {'verbose':0,'n_estimators':500, 'max_depth':4, 'learning_rate':0.2},       #5
        {'verbose':0,'n_estimators':50, 'max_depth':4},
        {'verbose':0,'n_estimators':50, 'max_depth':3},
    ]
    levels = [3,4,2,1,10.20]


    pool = multiprocessing.Pool(processes=10)
    result = []

    choris = [
        (1001, datas[0],levels[0], lParams[0], ),
        (1002, datas[0],levels[0], lParams[1], ),
        (1003, datas[0],levels[0], lParams[2], ),
        (1004, datas[0],levels[0], lParams[3], ),
        (1005, datas[0],levels[0], lParams[4], ),
        (1006, datas[0],levels[0], lParams[5], ),
        (1007, datas[0],levels[0], lParams[6], ),
        (1008, datas[0],levels[0], lParams[7], ),
        (2001, datas[1],levels[0], lParams[0], ),
        ]
    for each in choris:
        #result.append(one_work(each[0], each[1], each[2], each[3]))
        pool.apply_async(one_work, each)
    pool.close()
    pool.join()
    for each in result:
        print each.get()
if __name__ == '__main__':
    main()


