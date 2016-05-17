
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

def get_feat_names(df):
    #return ["ta_adx14", "ta_mid14", "ta_pid14"]
    #return ["ta_adx_14","ta_adxr_14","ta_mdi_14","ta_pdi_14",\
    #        "ta_apo_12_26_0","ta_aroon_up_14","ta_aroon_down_14",\
    #        "ta_ad",\
    #        "diff_0_1","diff_1_1","diff_2_1","diff_3_1",\
    #        "ta_aroonosc_14",\
    #        "ta_adsoc","ta_obv","ta_atr_14","ta_natr_14","ta_trange"]
    return [x for x in df.columns if x.startswith('ta_')]
def get_label_name(df, level):
    return "label3"


def build_trains(sym2feats, start, end):
    dfTrains = merge(sym2feats, start ,end).dropna()
    return dfTrains
    #    df = sym2feats[key].dropna()
    #    npCur = df.loc[start:end,get_feat_names(df)].values
    #    npCurLabel = df.loc[start:end, ["label8"]].values
    #    assert len(get_feat_names(df)) == npCur.shape[1]
    #    if npTrains is None:
    #        npTrains = npCur
    #        npLabels = npCurLabel
    #    else:
    #        npTrains = np.vstack((npTrains, npCur))
    #        npLabels = np.vstack((npLabels, npCurLabel))
    #        assert len(get_feat_names(df)) == npTrains.shape[1]
    #return npTrains, npLabels.ravel()


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
            print dfMerged.shape
            dfMerged = dfMerged.append(df)
            print dfMerged.shape
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


def train(sym2feats, level, start1, end1, start2, end2):
    dfTrain = build_trains(sym2feats, start1, end1)
    dfTest = build_trains(sym2feats, start2, end2)
    model = GradientBoostingRegressor(n_estimators=50,learning_rate=0.1, max_depth=5, verbose=1)
    model.fit(dfTrain.loc[:,get_feat_names(dfTrain)].values, dfTrain.loc[:,get_label_name(dfTrain,level)].values)
    dfTest["pred"] = model.predict(dfTest.loc[:,get_feat_names(dfTrain)].values)

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

def main():
    sym2feats = get_all()
    #dfMerged = merge(sym2feats, '2014-01-01', '2099-01-01')
    #for feat in ["feat_three_outside_move_builder", "feat_three_inside_strike_builder", 'feat_three_star_south_builder', 'feat_three_ad_white_soldier_builder', 'feat_abandoned_baby_builder', 'feat_three_ad_block_builder', 'feat_belt_hold_builder', 'feat_break_away_builder', 'feat_conceal_baby_builder']:
    #    cal_cor(dfMerged, feat, 100, 0, "label1")
    #    cal_cor(dfMerged, feat, 100, 0, "label2")
    #    cal_cor(dfMerged, feat, 100, 0, "label5")
    #    cal_cor(dfMerged, feat, 100, 0, "label10")
    #    cal_cor(dfMerged, feat, 100, 0, "label30")
    #sys.exit(0)

    #dfMerged = merge(sym2feats, '2016-05-01', '2099-01-01')
    #print dfMerged.head()
    #dfMerged.to_csv('merged-2016-05-01.csv')
    #npMergedFeat = dfMerged.loc[:,get_feat_names(dfMerged)].values
    #model = train(sym2feats, '2006-05-01', '2016-05-01', '2006-05-01', '2016-05-01')
    #npMergedPred = model.predict(npMergedFeat)
    #dfMerged["pred"] = npMergedPred
    #dfMerged.to_csv('pred-2016-05-01.csv')
    dfTestAll = None
    for level in (3,4):
        #start1, end1, start2, end2 = '2004-01-01', '2014-01-01', '2014-01-01', '2015-01-01'
        #dfTest = train(sym2feats, level, start1, end1, start2, end2); 
        #dfTest.to_csv(os.path.join(local_path, '..', 'data', 'pred', 'pred_%d_%s_%s_%s_%s.csv'%(level, start1, end1, start2, end2)))
        #start1, end1, start2, end2 = '2003-01-01', '2013-01-01', '2013-01-01', '2014-01-01'
        #dfTest = train(sym2feats, level, start1, end1, start2, end2); 
        #dfTest.to_csv(os.path.join(local_path, '..', 'data', 'pred', 'pred_%d_%s_%s_%s_%s.csv'%(level, start1, end1, start2, end2)))
        #start1, end1, start2, end2 = '2002-01-01', '2012-01-01', '2012-01-01', '2013-01-01'
        #dfTest = train(sym2feats, level, start1, end1, start2, end2); 
        #dfTest.to_csv(os.path.join(local_path, '..', 'data', 'pred', 'pred_%d_%s_%s_%s_%s.csv'%(level, start1, end1, start2, end2)))
        #start1, end1, start2, end2 = '2001-01-01', '2011-01-01', '2011-01-01', '2012-01-01'
        #dfTest = train(sym2feats, level, start1, end1, start2, end2); 
        #dfTest.to_csv(os.path.join(local_path, '..', 'data', 'pred', 'pred_%d_%s_%s_%s_%s.csv'%(level, start1, end1, start2, end2)))
        #start1, end1, start2, end2 = '2000-01-01', '2010-01-01', '2010-01-01', '2011-01-01'
        #dfTest = train(sym2feats, level, start1, end1, start2, end2); 
        #dfTest.to_csv(os.path.join(local_path, '..', 'data', 'pred', 'pred_%d_%s_%s_%s_%s.csv'%(level, start1, end1, start2, end2)))
        dfTest = train(sym2feats, level, '2004-01-01', '2014-01-01', '2014-01-01', '2015-01-01');
        dfTest = train(sym2feats, level, '2003-01-01', '2013-01-01', '2013-01-01', '2014-01-01'); dfTestAll = dfTestAll.append(dfTest)
        dfTestAll.to_csv(os.path.join(local_path, '..', 'data', 'pred', 'pred_%d.csv'%(level)))
        #dfTest = train(sym2feats, level, '2002-01-01', '2012-01-01', '2012-01-01', '2013-01-01'); dfTestAll = dfTestAll.append(dfTest)
        #dfTest = train(sym2feats, level, '2001-01-01', '2011-01-01', '2011-01-01', '2012-01-01'); dfTestAll = dfTestAll.append(dfTest)
        #dfTest = train(sym2feats, level,'2000-01-01', '2010-01-01', '2010-01-01', '2011-01-01');  dfTestAll = dfTestAll.append(dfTest)

    
    #print '2002-01-01', '2012-01-01', '2012-06-01', '2013-01-01'
    #train(sym2feats, '2002-01-01', '2012-01-01', '2012-06-01', '2013-01-01')
    #print '2002-01-01', '2012-01-01', '2012-01-01', '2012-06-01'
    #train(sym2feats, '2002-01-01', '2012-01-01', '2012-01-01', '2012-06-01')
    #print '2003-01-01', '2013-01-01', '2013-01-01', '2014-01-01'
    #train(sym2feats, '2003-01-01', '2013-01-01', '2013-01-01', '2014-01-01')
    #print '2004-01-01', '2014-01-01', '2014-01-01', '2015-01-01'
    #train(sym2feats, '2004-01-01', '2014-01-01', '2014-01-01', '2015-01-01')




if __name__ == '__main__':
    main()


