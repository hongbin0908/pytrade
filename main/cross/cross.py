#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os
import json
from sklearn.externals import joblib # to dump model
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
    print impstr
    exec impstr

    dfAll = None
    for each in conf.l_params:
        print each
        cls = joblib.load(os.path.join(root, 'data', 'models',"model_" + each[0]+ ".pkl"))
        sym2ta = model.get_all_from(each[1])
        df = model.build_trains(sym2ta, each[3][0], each[3][1])
        feat_names = model.get_feat_names(df)
        npFeat = df.loc[:,feat_names].values
        df["pred"] = cls.predict_proba(npFeat)[:,1]
        if dfAll is None:
            dfAll = df
        else:
            dfAll = dfAll.append(df)
    dfAll.to_csv(os.path.join(root, 'data', 'crosses', conf_file+ ".csv"))
    dacc =  accu(dfAll, each[2], 0.5)
    with open(os.path.join(root, "data", "crosses", conf_file + ".acc"), 'w') as fresult:
        print >> fresult, json.dumps(dacc)

if __name__ == '__main__':
    main(sys.argv)
    



