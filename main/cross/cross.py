#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os
from sklearn.externals import joblib # to dump model
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)
sys.path.append(local_path)

import model.modeling as  model

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



if __name__ == '__main__':
    main(sys.argv)
    



