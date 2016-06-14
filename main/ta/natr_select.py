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




def get_ta(taName):
    return pd.read_hdf(os.path.join(root, 'data', taName, 'merged.pkl'), "df")
def main(argv):
    ta = get_ta("ta1")
    print ta[['sym', 'date','open', 'high','low','close', 'ta_natr_14']].query('ta_natr_14 < 0.9 and ta_natr_14 > 0.8').count()
    print ta[['sym', 'date','open', 'high','low','close', 'ta_natr_14']].query('ta_natr_14 < 1.0 and ta_natr_14 > 0.9').count()
    print ta[['sym', 'date','open', 'high','low','close', 'ta_natr_14']].query('ta_natr_14 < 0.8').count()
    print ta[['sym', 'date','open', 'high','low','close', 'ta_natr_14']].count()
if __name__ == '__main__':
    main(sys.argv)
    
