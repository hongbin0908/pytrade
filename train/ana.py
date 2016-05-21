
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author 
import sys,os
import pandas as pd
from sklearn import metrics
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
import model_base as base

    
def accu(npTestLabel, npPred, threshold):
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
def main():
    for level in (3,4):
        print "===============level %d =================" % level
        df = pd.read_csv(os.path.join(root, 'data', 'pred', "pred_%s.csv" % level), index_col = ['date','sym'])
        print metrics.mean_absolute_error(df["label%d"%level].values, df["pred"].values)
        
        accu(df["label%d"%level].values, df["pred"].values, 0.0)
        accu(df["label%d"%level].values, df["pred"].values, 0.02)
        accu(df["label%d"%level].values, df["pred"].values, 0.03)
        accu(df["label%d"%level].values, df["pred"].values, 0.04)
        
if __name__ == '__main__':
    main()


