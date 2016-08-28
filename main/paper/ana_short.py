#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

import sys,os
import json
import numpy as np
import pandas as pd
from sklearn.externals import joblib # to dump model

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

import main.base as base
import main.ta as ta

def main(argv):
    infile = argv[0]
    top = int(argv[1])
    thresh = int(argv[2])
    df = pd.read_csv(infile)
    df['yyyy'] = df.date.str.slice(0,4)
    df["yyyyMM"] = df.date.str.slice(0,7)
    dfTrue = df[df["label5"] < 1.0]
    df2 = df.sort_values(["pred"], ascending=True).head(thresh).groupby('date').head(top)
    df2True = df2[df2["label5"] < 1.0]
    re = \
        df.groupby('yyyy').count().join(dfTrue.groupby('yyyy').count(), rsuffix='_dfTrue').join(df2.groupby('yyyy').count(),rsuffix='_df2').join(df2True.groupby('yyyy').count(),
             rsuffix='_df2True')[['pred','pred_dfTrue', 'pred_df2','pred_df2True']]
    re["rate1"] = re["pred_dfTrue"]*1.0/re["pred"]
    re["rate2"] = re["pred_df2True"]*1.0/re["pred_df2"]
    print re
    
if __name__ == '__main__':
    main(sys.argv[1:])
