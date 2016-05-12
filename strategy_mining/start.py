
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author 
import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")
from model_base import *
from sklearn import metrics
from sklearn import linear_model
from sklearn import tree
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import neural_network
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn import preprocessing

def get_feat_names():
    l = []
    for i in range(0, 76):
        l.append("feat"+str(i))
    return l
def build_trains(sym2feats, start, end):
    npTrains = None
    npLabels = None
    for key in sym2feats.keys():
        df = sym2feats[key].dropna()
        npCur = df.loc[start:end,get_feat_names()].values
        npCurLabel = df.loc[start:end, ["label"]].values
        assert len(get_feat_names()) == npCur.shape[1]
        if npTrains is None:
            npTrains = npCur
            npLabels = npCurLabel
        else:
            npTrains = np.vstack((npTrains, npCur))
            npLabels = np.vstack((npLabels, npCurLabel))
            assert len(get_feat_names()) == npTrains.shape[1]
    print npTrains.shape, npLabels.shape
    return npTrains, npLabels.ravel()



def ana(npTestLabel, npPred, threshold):
    npPos = npPred[npPred >= 1+threshold]
    npTrueInPos = npTestLabel[(npPred >= 1.0+threshold) & (npTestLabel>=1.0)]
    npNeg = npPred[npPred < 1-threshold]
    npFalseInNeg = npTestLabel[(npPred < 1.0-threshold) & (npTestLabel< 1.0)]
    npTrue = npTestLabel[npTestLabel >= 1.0]
    print "%d\t%d\t%f\t" % (npPos.size, npTrueInPos.size, npTrueInPos.size*1.0/npPos.size),
    print "%d\t%d\t%f\t" % (npNeg.size, npFalseInNeg.size, npFalseInNeg.size*1.0/npNeg.size),
    print "%d\t%d\t%f\t" % (npTrue.size, npTestLabel.size, npTrue.size*1.0/npTestLabel.size)

def merge(sym2feats, start ,end):
    dfMerged = None
    for sym in sym2feats.keys():
        df = sym2feats[sym]
        df = df.loc[start:end,:]
        index2 = df.index.values
        index1 = np.empty_like(index2, dtype='string')
        for i in range(0,index1.size):
            index1[i] = sym
        df = pd.DataFrame(df.values, index = [index1, index2], columns = df.columns.values )
        df.index.names = ['sym','date']
        if dfMerged is None:
            dfMerged = df
        else:
            dfMerged = dfMerged.append(df)

    return dfMerged
def main():
    sym2feats = get_all()
    npTrainFeat, npTrainLabel = build_trains(sym2feats, '2000-01-01','2012-01-01')
    npTestFeat, npTestLabel = build_trains(sym2feats, '2012-01-01','2099-01-01')
    model = GradientBoostingRegressor(n_estimators=40)
    print npTrainLabel
    model.fit(npTrainFeat, npTrainLabel)
    npPred = model.predict(npTestFeat)
    ana(npTestLabel, npPred, 0.0)
    ana(npTestLabel, npPred, 0.01)
    ana(npTestLabel, npPred,0.005)
    ana(npTestLabel, npPred, 0.02)
    ana(npTestLabel, npPred, 0.05)
    ana(npTestLabel, npPred, 0.08)
    ana(npTestLabel, npPred, 0.10)
    dfMerged = merge(sym2feats, '2016-01-01', '2099-01-01')
    npMergedFeat = dfMerged.loc[:,get_feat_names()].values
    npMergedPred = model.predict(npMergedFeat)
    dfMerged["pred"] = npMergedPred
    ana(dfMerged["label"].values, dfMerged['pred'].values)
    dfMerged.loc[:, ["open",'high','low','close','volume','adjclose','label','pred']].to_csv('pred.csv')



if __name__ == '__main__':
    main()


